'''
Requires numpy, scipy, sklearm
iris needs to be installed (it is required by other modules within this package
xarray cubes should work via iris interface
'''
import logging
import re
import ntpath
import os

import iris
import iris.coord_categorisation as icc
import iris.util as iutil
import numpy as np

obs_2_default_path = {'ESA_satellite_0p25': '/work/scratch-pw2/schan016/NOC-hostace/SST-10x10/',
                      'ESA_satellite_0p25-mid': '/work/scratch-pw2/schan016/NOC-hostace/SST-10x10-mid-scale/',
                      'ESA_satellite_0p25-ukmo': '/work/scratch-pw2/schan016/NOC-hostace/SST-10x10-UKMO_grid/',
                      'ESA_satellite_1p00-daily': '/work/scratch-pw2/schan016/NOC-hostace/SST-10x10_daily/',
                      'ESA_satellite_5p00-monthly': '/work/scratch-pw2/schan016/NOC-hostace/ESA_CCI5deg_month/ANOMALY/',
                      'ERA5_T2M_0p25': '/work/scratch-pw2/schan016/NOC-hostace/ERA_T2M-10x10/TenDeg/',
                      'ERA5_T2M_0p25-daily': '/work/scratch-pw2/schan016/NOC-hostace/ERA_T2M-10x10-daily/TenDeg/',
                      'ERA5_T2M_0p25-ukmo-h': '/work/scratch-pw2/schan016/NOC-hostace/ERA_T2M_regrid/ANOMALY_hres_pent_regridded/10x10/',
                      'ERA5_T2M_0p25-ukmo-0': '/work/scratch-pw2/schan016/NOC-hostace/ERA_T2M_regrid/ANOMALY_ens0_pent_regridded/10x10/',
                      'ERA5_T2M_0p25-ukmo-1': '/work/scratch-pw2/schan016/NOC-hostace/ERA_T2M_regrid/ANOMALY_ens1_pent_regridded/10x10/',
                      'ERA5_T2M_0p25-ukmo-m': '/work/scratch-pw2/schan016/NOC-hostace/ERA_T2M_regrid/ANOMALY_ensmean_pent_regridded/10x10/'}

'''
Definition of "ocean basins"
The ranges are not strictly for those basin, but just a way to break the globe up into different
lego pieces
For instance
"Atlantic Ocean" below contains bits of SE Pacific
"Indian Ocean" below does not reach land of kangroos and koalas
'''
''' lon: -180 - 180 '''
basin_2_limit = {'Atlantic_Ocean': {'latitude': (-60, 60), 'longitude': (-100, 20)},
                 'Indian_Ocean': {'latitude': (-60, 30), 'longitude': (20, 110)}, ## Original (60S-30N Indian Ocean) domain
                 'Indian_Ocean_X': {'latitude': (-60, 60), 'longitude': (20, 110)}, ## Full latitude band
                 'Indian_Ocean_H': {'latitude': (30, 60), 'longitude': (20, 110)}, ## Missing northern part of Indian_Ocean
                 'Pacific_Ocean': {'latitude': (-60, 60), 'longitude': (110, -100)},
                 'Global_No_Poles': {'latitude': (-60, 60), 'longitude': (-180, 180)},
                 'Global_With_Poles': {'latitude': (-90, 90), 'longitude': (-180, 180)},
                 'Southern_Ocean': {'latitude': (-80, -60), 'longitude': (-180, 180)},
                 'Southern_Ocean_X': {'latitude': (-90, -60), 'longitude': (-180, 180)},
                 'Arctic_Ocean': {'latitude': (60, 80), 'longitude': (-180, 180)},
                 'Arctic_Ocean_X': {'latitude': (60, 90), 'longitude': (-180, 180)}}

''' lon: 0 - 360 '''
basin_2_limit_era = {'Atlantic_Ocean': {'latitude': (-60, 60), 'longitude': (260, 20)},
                     'Indian_Ocean': {'latitude': (-60, 30), 'longitude': (20, 110)}, ## Original (60S-30N Indian Ocean) domain
                     'Indian_Ocean_X': {'latitude': (-60, 60), 'longitude': (20, 110)}, ## Full latitude band
                     'Indian_Ocean_H': {'latitude': (30, 60), 'longitude': (20, 110)}, ## Missing northern part of Indian_Ocean
                     'Global_No_Poles': {'latitude': (-60, 60), 'longitude': (-180, 180)},
                     'Pacific_Ocean': {'latitude': (-60, 60), 'longitude': (110, 260)},
                     'Southern_Ocean': {'latitude': (-80, -60), 'longitude': (0, 360)},
                     'Southern_Ocean_X': {'latitude': (-90, -60), 'longitude': (0, 360)},
                     'Arctic_Ocean': {'latitude': (60, 80), 'longitude': (0, 360)},
                     'Arctic_Ocean_X': {'latitude': (60, 90), 'longitude': (0, 360)}}

month_2_constraint = {(month_num+1): iris.Constraint(month_number=month_num+1) for month_num in range(12)}

'''
Starting with those 10 deg box netcdfs, say data_160_0.nc (160E-170E, 0-10N)
identify the netcdfs that are n_10box away from it, including diagonal ones
n_10box == 1: total of  9 files (the original one plus the 8 around it)
n_10box == 2: total of 25 files

See "R_scripts", that is the original script that breaks up the larger ERA and ESA CCI files
into 10 degree boxes
'''
def from_one_to_nine_data_lat_lon_convention(ncfile0, n_10box = 1, lat_max = 70):
    #ncfile0_path = ntpath.dirname(ncfile0)
    #if len(ncfile0_path) > 0:
    #    ncfile0_path = ncfile0_path+'/'
    #else:
    #    ncfile0_path = './'
    ncfile0_basename = ntpath.basename(ncfile0)
    match0 = re.findall(r'[^\s_]+', ncfile0_basename)
    match1 = re.findall(r'[^\s.]+', match0[-1])
    lon, lat = int(match0[1]), int(match1[0])
    lon_b = [lon-10*n for n in range(1, n_10box+1)]
    lon_n = [lon+10*n for n in range(1, n_10box+1)]
    lat_b = [lat-10*n for n in range(1, n_10box+1)]
    lat_n = [lat+10*n for n in range(1, n_10box+1)]
    lon_b = [(llon+360) if (llon <  -180) else llon for llon in lon_b]
    lon_n = [(llon-360) if (llon >=  180) else llon for llon in lon_n]
    ##
    ans = []
    for x in lon_b+[lon]+lon_n:
        for y in lat_b+[lat]+lat_n:
            if (y < lat_max) and (y > (-lat_max-10)):
                ans.append('data_'+str(x)+'_'+str(y)+'.nc')
    ans.sort()
    return ans

def from_one_to_nine_data_lat_lon_convention_era(ncfile0, n_10box = 1, lat_max = 70):
    ncfile0_basename = ntpath.basename(ncfile0)
    match0 = re.findall(r'[^\s_]+', ncfile0_basename)
    match1 = re.findall(r'[^\s.]+', match0[-1])
    lon, lat = int(match0[1]), int(match1[0])
    lon_b = [lon-10*n for n in range(1, n_10box+1)]
    lon_n = [lon+10*n for n in range(1, n_10box+1)]
    lat_b = [lat-10*n for n in range(1, n_10box+1)]
    lat_n = [lat+10*n for n in range(1, n_10box+1)]
    lon_b = [(llon+360) if (llon <    0) else llon for llon in lon_b]
    lon_n = [(llon-360) if (llon >= 360) else llon for llon in lon_n]
    ##
    ans = []
    for x in lon_b+[lon]+lon_n:
        for y in lat_b+[lat]+lat_n:
            if (y < lat_max) and (y > (-lat_max-10)):
                ans.append('data_'+str(x)+'_'+str(y)+'.nc')
    ans.sort()
    return ans

'''
Identify 10 deg box netcdfs files within the definitions in dictionary basin_2_limit
'''
def default_files_in_basin(basin_name = 'Indian_Ocean'):
    lat_bounds = basin_2_limit[basin_name]['latitude']
    lon_bounds = basin_2_limit[basin_name]['longitude']
    lats = np.arange(lat_bounds[0], lat_bounds[1], 10).tolist()
    if lon_bounds[1] > lon_bounds[0]:
        lons = np.arange(lon_bounds[0], lon_bounds[1], 10).tolist()
    else:
        ''' wrapped around 180E/W '''
        lons1 = np.arange(lon_bounds[0],  180, 10).tolist()
        lons2 = np.arange(-180, lon_bounds[1], 10).tolist()
        lons  = np.concatenate((lons1, lons2), axis = None)
    ans = []
    for lat in lats:
        for lon in lons:
            ans.append('data_'+str(lon)+'_'+str(lat)+'.nc')
    return ans

def default_files_in_basin_era(basin_name = 'Indian_Ocean'):
    lat_bounds = basin_2_limit_era[basin_name]['latitude']
    lon_bounds = basin_2_limit_era[basin_name]['longitude']
    lats = np.arange(lat_bounds[0], lat_bounds[1], 10).tolist()
    if lon_bounds[1] > lon_bounds[0]:
        lons = np.arange(lon_bounds[0], lon_bounds[1], 10).tolist()
    else:
        ''' wrapped around 180E/W '''
        lons1 = np.arange(lon_bounds[0], 360, 10).tolist()
        lons2 = np.arange(  0, lon_bounds[1], 10).tolist()
        lons  = np.concatenate((lons1, lons2), axis = None)
    ans = []
    for lat in lats:
        for lon in lons:
            ans.append('data_'+str(lon)+'_'+str(lat)+'.nc')
    return ans

'''
iris.load wrapper
'''
def iris_load_cube_plus(ncfiles,
                        var_name='sst_anomaly',
                        fudge_negative_longitude=False,
                        fudge_360plus_longitude=False,
                        callback=None,
                        additional_constraints=None):
    #
    cubes = iris.load(ncfiles, var_name, callback=callback)
    for cube in cubes:
        assert 'time' in [coord.name() for coord in cube.coords()]
    #
    if fudge_negative_longitude and fudge_360plus_longitude:
        raise ValueError('fudge_negative_longitude and fudge_360plus_longitude are mutually exclusive')
    if fudge_negative_longitude:
        print('fudge_negative_longitude is True')
        for cube_i, cube in enumerate(cubes):
            cubes[cube_i].coord('longitude').bounds = None
            #cube.coord('longitude').points.setflags(write = 1)
            #negative_lons_idx = np.where(cube.coord('longitude').points < 0)
            #cube.coord('longitude').points[negative_lons_idx] = cube.coord('longitude').points[negative_lons_idx] + 180
            #cube.coord('longitude').points.setflags(write = 0)
            any_negative_idx = np.any(cube.coord('longitude').points < 0)
            if any_negative_idx:
                print('Modifying lons in :', cubes[cube_i].coord('longitude').points[0])
                cubes[cube_i].coord('longitude').points = cubes[cube_i].coord('longitude').points + 360.0
    #
    if fudge_360plus_longitude:
        print('fudge_360plus_longitude is True')
        for cube_i, cube in enumerate(cubes):
            cubes[cube_i].coord('longitude').bounds = None
            cube_lons = cube.coord('longitude').points
            any_wraparound_idx = np.logical_and(cube_lons[-1] <= 180.0,
                                                cube_lons[0] >= 0.0)
            # any_wraparound_idx = np.logical_and(np.any(cube.coord('longitude').points <= 180.0),
            #                                     np.any(cube.coord('longitude').points >=   0.0))
            if any_wraparound_idx:
                print('Modifying lons:', cube_lons[0], cube_lons[-1])
                new_cube_lons = cube_lons + 360.0
                print('New lons: ', new_cube_lons)
                cubes[cube_i].coord('longitude').points = new_cube_lons
    #
    iutil.unify_time_units(cubes)
    # Probably not needed anyway
    # how to call it depends on iris version
    try:
        iutil.equalise_attributes(cubes)
    except Exception as e:
        print('The following Python exception detected:')
        logging.error(repr(e))
    ans_cube = cubes.concatenate_cube()
    #
    for t_meta_gen_func in [icc.add_month_number,
                            icc.add_day_of_month,
                            icc.add_year]:
        try:
            t_meta_gen_func(ans_cube, 'time')
        except Exception as e:
            print('Python exception detected (extra t coord probably exist):')
            logging.error(repr(e))
    ##
    if additional_constraints is not None:
        ans_cube = ans_cube.extract(additional_constraints)
    ##
    return ans_cube

#
# Masking tool for 1 deg?
#
fname = iris.sample_data_path('air_temp.pp')
air_temp = iris.load_cube(fname)
cube_projection = air_temp.coord('latitude').coord_system
def add_proj_fix_mask_callback_ESA_1deg(cube, field, filename):
    cube.coord('lat').units = 'degrees'
    cube.coord('lon').units = 'degrees'
    cube.coord('lat').rename('latitude')
    cube.coord('lon').rename('longitude')
    cube.coord('latitude').coord_system = cube_projection
    cube.coord('longitude').coord_system = cube_projection
    cube.coord('latitude').guess_bounds()
    cube.coord('longitude').guess_bounds()

    
def getmask_ESA_with_ice_1deg():
    # Includes ice-covered grid points
    mask_path = '/gws/nopw/j04/hostace/schan016/ESA_CCI1deg_daily/land_sea_mask/'
    wor_mask = iris.load_cube(mask_path+'world_landmask.nc',
                              callback=add_proj_fix_mask_callback_ESA_1deg)
    wor_mask.data = wor_mask.data.astype('int8')
    wor_mask.rename('land_sea_mask')
    return wor_mask


def getmask_ESA_1deg(threshold = 0.8):
    # Pure land-sea mask
    esa_cubes = iris.load('/gws/nopw/j04/hostace/schan016/ESA_CCI1deg_daily/ANOMALY/1982/19820101_regridded_sst.nc')
    sea_area_fraction = esa_cubes.extract('sea_area_fraction')[0][0]
    ls_mask = sea_area_fraction.copy()
    ls_mask.data[sea_area_fraction.data >= threshold] = 1
    ls_mask.data[sea_area_fraction.data <  threshold] = 0
    ls_mask.data = ls_mask.data.astype('int8')
    ls_mask.rename('land_sea_mask')
    return ls_mask


def mask_cube_ESA_1deg_with_ice(cube2mask):
    #
    wor_mask = getmask_ESA_with_ice_1deg()
    #
    lat_rng = (np.floor(cube2mask.coord('latitude').points.min()),
               np.ceil (cube2mask.coord('latitude').points.max()))
    lon_rng = (np.floor(cube2mask.coord('longitude').points.min()),
               np.ceil (cube2mask.coord('longitude').points.max()))
    latc = iris.Constraint(latitude  = lambda val: (lat_rng[1]+0.05) > val > (lat_rng[0]-0.05))
    lonc = iris.Constraint(longitude = lambda val: (lon_rng[1]+0.05) > val > (lon_rng[0]-0.05))
    submask = wor_mask.extract(latc & lonc)
    #
    mask2 = np.broadcast_to(submask.data, cube2mask.data.shape)
    cube2mask.data = np.ma.masked_where(mask2 == 0.0, cube2mask.data)
    return cube2mask


def mask_cube_ERA_0p25(cube2mask, use_alternative=False):
    # This is the original path in CEDA for
    # ecmwf-era5_oper_an_sfc_200001010000.lsm.inv.nc
    # _lsmask_file_era_path = '/badc/ecmwf-era5/data/invariants/'
    _lsmask_file_era_path = os.path.dirname(__file__)+'/data/'
    #
    # use_alternative = True or use_alternative = False both work
    # However, cube2mask must be checked to make sure for longitude wrap around/discontinuity
    # cube.intersection does not work properly with them if there are longitude jumps
    #
    # Below are situations that will fail:
    # 350 (10W) --> 0 or 360 (0E/W) --> 10 (10E)
    # 170 (170E) --> (-)180 (180E/W) --> -170 (170W)
    #
    # What will work:
    # 350 --> 360 --> 370 or  -10 --> 0 --> 10
    # 170 --> 180 --> 190
    #
    if use_alternative:
        _lsmask_file_era = _lsmask_file_era_path+'ecmwf-era5_oper_an_sfc_200001010000.lsm.inv.alternative.nc'
    else:
        _lsmask_file_era = _lsmask_file_era_path+'ecmwf-era5_oper_an_sfc_200001010000.lsm.inv.nc'
    _lsmask_cube_era = iris.load_cube(_lsmask_file_era)[0]
    lons = cube2mask.coord('longitude').points
    lats = cube2mask.coord('latitude').points
    lsmask_cube_regional = _lsmask_cube_era.intersection(longitude=(lons[0], lons[-1]), 
                                                         latitude=(lats[0], lats[-1]))
    lsmask_cube_regional = iutil.reverse(lsmask_cube_regional, 'latitude')
    print(repr(lsmask_cube_regional), lsmask_cube_regional.shape)
    mask2 = np.broadcast_to(lsmask_cube_regional.data, cube2mask.data.shape)
    print(mask2.shape)
    cube2mask.data = np.ma.masked_where(mask2 == 1, cube2mask.data)
    return cube2mask


#
#

def _test_load(var_name = 'sst_anomaly'):
    import glob
    import random
    ncfiles = glob.glob(obs_2_default_path['ESA_satellite_0p25']+'data*.nc')
    for i in range(10):
        ncfile0 = random.choice(ncfiles)
        ans = from_one_to_nine_data_lat_lon_convention(ncfile0, lat_max = 80)
        print(ncfile0, ans)
        ncfiles = [obs_2_default_path['ESA_satellite_0p25']+ncfile for ncfile in ans]
        cube = iris_load_cube_plus(ncfiles)
        print(cube)
    return

def main():
    print('=== Main ===')
    _test_load()
    return

if __name__ == "__main__":
    main()
