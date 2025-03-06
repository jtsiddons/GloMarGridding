'''
Run script for SST and LSAT nonstationary variogram parameter estimation
'''
# import datetime
import logging
from pathlib import Path
import sys

import iris
from iris import coord_categorisation as icc
from iris.fileformats import netcdf as inc
from iris.util import equalise_attributes
import numpy as np
import yaml

import psutil

from ellipse_estimation import cube_covariance
from glomar_gridding.utils import init_logging

# pylint: disable=logging-fstring-interpolation

with open('process_basin_satellite_monthly_climatology_matern_physical_distances_Global.yaml', 'r') as f:
    fit_intel = yaml.safe_load(f)

def load_sst():
    ''' Load SST inputs '''
    ncfile = fit_intel['sst_in']
    cube = iris.load_cube(ncfile)
    icc.add_month_number(cube, 'time', name='month_number')
    return cube


def load_lsat():
    ''' Load LSAT inputs '''
    ncfile = fit_intel['lsat_in']
    cube = iris.load_cube(ncfile, 'land 2 metre temperature')
    icc.add_month_number(cube, 'time', name='month_number')
    return cube


def mask_time_union(cube):
    '''
    Make sure mask is same for all time
    If a single masking occur,
    masks all other time as well
    '''
    cube_mask = cube.data.mask
    common_mask = np.any(cube_mask, axis=0)
    cube.data.mask = common_mask
    return cube


def main():
    '''
    MAIN
    '''
    #
    init_logging(level="WARN")
    #
    logging.info('Start')
    #
    print(psutil.Process().cpu_affinity())
    nCPUs = len(psutil.Process().cpu_affinity())
    print('len(cpu_affinity) = ', nCPUs)
    #
    dat_type = sys.argv[1]
    month_value = int(sys.argv[2])
    v = float(sys.argv[3])
    fform = sys.argv[4]
    everyother = 1
    # For Met Office-styled grid ::2 will sit on (y=*.5 x=*.5)
    # Since data is from 89.5S and 89.5N,
    # Southern Ocean_X cannot use offset for latitude because y = 0 is on lat = 89.5
    for argh, sysargv in enumerate(sys.argv):
        print('sys.argv['+str(argh)+'] = ', sysargv)
    #
    assert dat_type in ['sst', 'lsat'], 'dat_type (sys.argv[1]) must be sst or lsat'
    if dat_type == 'lsat':
        data_loader = load_lsat
        outpath_base = fit_intel['lsat_out_base']
        print('dat_type is lsat')
    else:
        data_loader = load_sst
        outpath_base = fit_intel['sst_out_base']
        print('dat_type is sst')
    #
    print('v = ', v)
    #
    additional_constraints = iris.Constraint(month_number=month_value)
    surftemp_cube = data_loader()
    surftemp_cube = surftemp_cube.extract(additional_constraints)
    surftemp_cube = mask_time_union(surftemp_cube)
    #
    surftemp_cube_time_length = len(surftemp_cube.coord('time').points)
    print(repr(surftemp_cube))
    #
    if fform == 'anistropic_rotated_pd':
        nparms = 6
    elif fform == 'anistropic_pd':
        nparms = 5
    elif fform == 'isotropic_pd':
        nparms = 4
    else:
        raise ValueError('Unknown fform')
    defval = [-999.9 for _ in range(nparms)]
    #
    # Init values set to HadCRUT5 defaults
    # no prior distrubtion set around those value
    init_values = (2000.0, 2000.0, 0)
    # Uniformative prior of parameter range
    fit_bounds = ((300.0, 30000.0),
                  (300.0, 30000.0),
                  (-2.0*np.pi, 2.0*np.pi))
    #
    super_cube_list = iris.cube.CubeList()
    #
    logging.info(repr(surftemp_cube))
    print(surftemp_cube.coord('latitude'))
    print(surftemp_cube.coord('longitude'))
    print(surftemp_cube.coord('time'))
    print('Large cube built for cov caculations:', repr(surftemp_cube))
    logging.info('Building covariance matrix')
    super_sst_cov = cube_covariance.CovarianceCube(surftemp_cube)
    logging.info('Covariance matrix completed')
    #
    sst_cube_not_template = surftemp_cube[surftemp_cube_time_length//2]
    for zonal, zonal_slice in enumerate(sst_cube_not_template.slices(['longitude'])):
        # Zonal slices
        logging.info(f"{zonal} {repr(zonal_slice)}")
        if (zonal % everyother) != 0:
            continue
        zonal_cube_list = iris.cube.CubeList()
        for box_count, invidiual_box in enumerate(zonal_slice.slices([])):
            #
            if (box_count % everyother) != 0:
                continue
            logging.info(f"{zonal} || {box_count} {repr(invidiual_box)}")
            #
            current_lon = invidiual_box.coord('longitude').points[0]
            current_lat = invidiual_box.coord('latitude').points[0]
            logging.info(f"{zonal} || {box_count} {current_lon} {current_lat}")
            if np.ma.is_masked(invidiual_box.data):
                xy, actual_latlon = super_sst_cov.find_nearest_xy_index_in_cov_matrix([current_lon, current_lat], use_full=True)
                kwargs = {'model_type': cube_covariance.fform_2_modeltype[fform],
                          'additional_meta_aux_coords': [cube_covariance.make_v_aux_coord(v)],
                          'default_values': defval}
                print(super_sst_cov.data_cube)
                print(super_sst_cov.data_cube.coord('latitude'))
                print(super_sst_cov.data_cube.coord('longitude'))
                template_cube = super_sst_cov._make_template_cube2((current_lon, current_lat))
                ans  = cube_covariance.create_output_cubes(template_cube, **kwargs)['param_cubelist']
                ansH = 'MASKED'
            else:
                # Nearest valid point
                xy, actual_latlon = super_sst_cov.find_nearest_xy_index_in_cov_matrix([current_lon, current_lat])
                # Note:
                # Possible cause for convergence in ENSO grid points; max_distance is originally
                # introduced to keep moving window fits consistent (i.e. always using 20x20 deg squares around
                # central gp). Now with global inputs this can be relaxed, and use of global inputs will
                # ensure correlations from far away grid points be accounted for <--- this cannot be
                # done for moving window fits.
                kwargs = {'v': v,
                          'fform': fform,
                          'guesses': init_values,
                          'bounds': fit_bounds,
                          'max_distance': 60.0,
                          'n_jobs': nCPUs}
                ansX = super_sst_cov.ps2006_kks2011_model(xy, **kwargs)
                ans = ansX['Model_as_1D_cube']
                ansH = (ansX['Model'].x, ansX['Model'].x[-1]*180.0/np.pi)
            ans_lon = ans[0].coord('longitude').points
            ans_lat = ans[0].coord('latitude').points
            logging.info(f"{zonal} || {box_count} {xy} {actual_latlon} {ans_lon} {ans_lat} {ansH}")
            for individual_ans in ans:
                zonal_cube_list.append(individual_ans)
                zonal_cube_list.concatenate()
        for zonal_ans_cube in zonal_cube_list:
            super_cube_list.append(zonal_ans_cube)
            equalise_attributes(super_cube_list)
    logging.info("Grid box loop is completed")
    equalise_attributes(super_cube_list)
    try:
        super_cube_list = super_cube_list.concatenate()
    except:
        pass
    #
    vstring = sys.argv[3].replace('.','p')
    vstring = '_v_eq_'+vstring
    #
    outpath = outpath_base+'matern_physical_distances'+vstring+'/'
    Path(outpath).mkdir(parents=True, exist_ok=True)
    #
    outncfilename = outpath+dat_type+'_'
    outncfilename += str(month_value).zfill(2)+'.nc'
    print('Results to be saved...')
    print(super_cube_list)
    logging.info(f"Saving results to {outncfilename}")
    inc.save(super_cube_list, outncfilename)
    logging.info('Completed')


if __name__ == "__main__":
    main()
