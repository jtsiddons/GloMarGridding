import datetime
from pathlib import Path
import sys

import iris
from iris import coord_categorisation as icc
from iris.fileformats import netcdf as inc
from iris.util import equalise_attributes
import numpy as np

import psutil

from nonstationary_cov import cube_covariance


def load_sst():
    ncfile = '/gws/nopw/j04/hostace/data/ESA_CCI_5deg_monthly_extra/ANOMALY/'
    ncfile += 'esa_cci_sst_5deg_monthly_1982_2022.nc'
    cube = iris.load_cube(ncfile)
    icc.add_month_number(cube, 'time', name='month_number')
    return cube


def load_lsat():
    ncfile = '/gws/nopw/j04/hostace/schan016/ERA_SAT_monthly/ANOMALY/'
    ncfile += 'era5_t2m_monthly_mean_1979_2023_5x5.nc'
    cube = iris.load_cube(ncfile, 'land 2 metre temperature')
    icc.add_month_number(cube, 'time', name='month_number')
    return cube


def mask_time_union(cube):
    cube_mask = cube.data.mask
    common_mask = np.any(cube_mask, axis=0)
    cube.data.mask = common_mask
    return cube


def main():
    #
    print(datetime.datetime.now(), 'Start')
    #
    # nCPUs = 2
    print(datetime.datetime.now(), psutil.Process().cpu_affinity())
    print(datetime.datetime.now(), 'len(cpu_affinity) = ', len(psutil.Process().cpu_affinity()))
    nCPUs = len(psutil.Process().cpu_affinity())
    print(datetime.datetime.now(), 'number of jobs threads = ', nCPUs)
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
        outpath_base = '/gws/nopw/j04/hostace/schan016/ERA_SAT_monthly/ANOMALY/SpatialScales/'
        print('dat_type is lsat')
    else:
        data_loader = load_sst
        outpath_base = '/gws/nopw/j04/hostace/data/ESA_CCI_5deg_monthly_extra/ANOMALY/SpatialScales/'
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
    init_values = (1300.0, 1300.0, 0)
    # Uniformative prior of parameter range
    fit_bounds = ((300.0, 30000.0),
                  (300.0, 30000.0),
                  (-2.0*np.pi, 2.0*np.pi))
    #
    super_cube_list = iris.cube.CubeList()
    #
    print(datetime.datetime.now(), repr(surftemp_cube))
    print(datetime.datetime.now(), surftemp_cube.coord('latitude'))
    print(datetime.datetime.now(), surftemp_cube.coord('longitude'))
    print(datetime.datetime.now(), surftemp_cube.coord('time'))
    print(datetime.datetime.now(), 'Large cube built for cov caculations:', repr(surftemp_cube))
    print(datetime.datetime.now(), 'Building covariance matrix')
    super_sst_cov = cube_covariance.CovarianceCube(surftemp_cube)
    print(datetime.datetime.now(), 'Covariance matrix completed')
    #
    sst_cube_not_template = surftemp_cube[surftemp_cube_time_length//2]
    for zonal, zonal_slice in enumerate(sst_cube_not_template.slices(['longitude'])):
        # Zonal slices
        print(datetime.datetime.now(), zonal, repr(zonal_slice))
        if (zonal % everyother) != 0:
            continue
        zonal_cube_list = iris.cube.CubeList()
        for box_count, invidiual_box in enumerate(zonal_slice.slices([])):
            #
            if (box_count % everyother) != 0:
                continue
            print(datetime.datetime.now(), zonal, '||', box_count, repr(invidiual_box))
            #
            current_lon = invidiual_box.coord('longitude').points[0]
            current_lat = invidiual_box.coord('latitude').points[0]
            print(datetime.datetime.now(), zonal, '||', box_count, current_lon, current_lat)
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
            print(datetime.datetime.now(), zonal, '||', box_count, xy, actual_latlon, ans_lon, ans_lat, ansH)
            for individual_ans in ans:
                zonal_cube_list.append(individual_ans)
                zonal_cube_list.concatenate()
        for zonal_ans_cube in zonal_cube_list:
            super_cube_list.append(zonal_ans_cube)
            equalise_attributes(super_cube_list)
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
    print('Saving results to ',outncfilename)
    inc.save(super_cube_list, outncfilename)
    print(datetime.datetime.now(), 'Completed')


if __name__ == "__main__":
    main()
