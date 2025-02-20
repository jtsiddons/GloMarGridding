'''
Training period 1982-2022 (same as 1 deg exercise)
'''
import glob

import iris
from iris.util import equalise_attributes, unify_time_units
from iris.fileformats import netcdf as inc
import numpy as np


def fix_meta(cubes):
    equalise_attributes(cubes)
    unify_time_units(cubes)


def fix_latlonmeta_callback(cube, field, filename):
    try:
        cube.coord('latitude').long_name = 'latitude'
    except:
        pass
    else:
        cube.coord('latitude').attributes = None
        cube.coord('longitude').long_name = 'longitude'
        cube.coord('longitude').attributes = None
        cube.coord('time').points = cube.coord('time').points.astype(np.float32)
        cube.coord('time').bounds = None


def cat_check(cubes):
    for a, b in zip(cubes[:-1], cubes[1:]):
        moo = iris.cube.CubeList()
        moo.append(a)
        moo.append(b)
        try:
            moo.concatenate_cube()
        except:
            print(a.coord('time'))
            print(b.coord('time'))
            raise ValueError('Concat has failed between the two above times')


def load_sst_cubes0(yyyys,
                    ice_check=True,
                    ice_check_kwargs={}):
    nc_path = '/work/scratch-pw2/schan016/NOC-hostace/ESA_CCI5deg_month/ANOMALY/'
    nc_files = []
    for yyyy in yyyys:
        nc_files += glob.glob(nc_path+str(yyyy)+'????_regridded_sst.nc')
    nc_files.sort()
    cubes = iris.load(nc_files,
                      ['sea_water_temperature_anomaly',
                       'sea_ice_area_fraction'],
                      callback=fix_latlonmeta_callback)
    sst = cubes.extract('sea_water_temperature_anomaly')
    fix_meta(sst)
    cat_check(sst)
    sst = sst.concatenate_cube()
    if ice_check:
        sic = cubes.extract('sea_ice_area_fraction')
        fix_meta(sic)
        sic = sic.concatenate_cube()
        sst = apply_ice_threshold(sst, sic, **ice_check_kwargs)
    sst.coord('time').guess_bounds()
    return sst


def apply_ice_threshold(sst_cube,
                        sic_cube,
                        sic_threshold=0.15):
    '''
    It seems to be standard to use 15% as SIC threshold to define 
    ice covered pixel:
    https://nsidc.org/data/soac/sea-ice-concentration
    Liz downloads of ESA SST
    HadCRUT5 etc.

    In the newly downloaded pre-1981 and 2022+ 5x5 ESA data,
    no SIC mask is used.

    We are not using pre-1981 but 2022 was included in revised
    scale estimates for 1 deg pentad data.

    args:
    sst_cube = sst
    sic_cube = sic
    
    kwargs
    sic_threshold
    '''
    if np.logical_or(np.any(sic_cube.data > 1.0),
                     np.any(sic_cube.data < 0.0)):
        raise ValueError('sic_cube does not appear to be sic; got sic and sst backwards?')
    sic_arr_ge_threshold = sic_cube.data >= sic_threshold
    sst_cube.data.mask[sic_arr_ge_threshold] = True
    return sst_cube


def prep_sst_data():
    sst = load_sst_cubes0(range(1982, 2023))
    outfile = '/gws/nopw/j04/hostace/data/ESA_CCI_5deg_monthly_extra/ANOMALY/'
    outfile += 'esa_cci_sst_5deg_monthly_1982_2022.nc'
    inc.save(sst, outfile, zlib=True)


def main():
    prep_sst_data()


if __name__ == "__main__":
    main()
