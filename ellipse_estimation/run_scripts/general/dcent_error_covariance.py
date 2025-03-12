'''
Converts Duo's error covariance into a more ingestiable format for kriging code
'''

import warnings

import iris
import xarray as xa
import numpy as np
import pandas as pd

def datetime64_to_yyyy(datetime64_vec):
    '''
    numpy Datetime64
    https://numpy.org/doc/stable/reference/arrays.datetime.html
    Section Basic DateTime
    0 is set to Unix epoch (00:00:00 UTC on 1 January 1970)
    [Y] for year
    '''
    return datetime64_vec.astype('datetime64[Y]').astype(int) + 1970


def time_check(xa_time, yyyy0=1850, yyyyf=2023):
    ''' Check xarray time to between yyyy0 and yyyyf '''
    return np.logical_and(yyyyf >= datetime64_to_yyyy(xa_time),
                          datetime64_to_yyyy(xa_time) >= yyyy0)


def lsat_error_covariance_maker(ensemble_mem,
                                savez_compressed=True,
                                yyyy0=1850,
                                yyyyf=2023):
    '''
    Create SST error covariance matrices
    Files are by ensemble member

    Each file has only 1 variable "sigma2"
    Unlike SST, sigma2 has a time dimension
    There are no off diagonal terms for error covariance over land

    Outputs will be the 1D diagonal terms only 
    (otherwise eats too much space and I/O time)
    '''
    nc_path = '/noc/mpoc/surface_data/DCENTv1/'
    nc_path += 'DCENT_uncertainities/LSAT_Uncertainty_nc/5x5/'
    nc_file = nc_path + 'DCENT_en_'+str(ensemble_mem)+'_LSAT_Uncertainty_reso_5.nc'
    print('Reading '+nc_file)
    xa_cubes = xa.open_dataset(nc_file)
    # ir_cubes = iris.load(nc_file)
    #
    outpath = nc_path+'error_covariances/'
    if savez_compressed:
        out_error_covariance_npy = outpath+'lsat_error_covariance_'+str(ensemble_mem)+'.npz'
    else:
        out_error_covariance_npy = outpath+'lsat_error_covariance_'+str(ensemble_mem)+'.npy'
    #
    # ir_sampling_rnd_sigma_2 = ir_cubes[0]
    # print(repr(ir_sampling_rnd_sigma_2))
    #
    def timecheck2(tttt):
        return time_check(tttt, yyyy0=yyyy0, yyyyf=yyyyf)
    # timecheck2 = lambda tttt: time_check(tttt, yyyy0=yyyy0, yyyyf=yyyyf)
    xa_sampling_rnd_sigma_2 = xa_cubes['sigma2']
    xa_sampling_rnd_sigma_2 = xa_sampling_rnd_sigma_2[timecheck2(xa_sampling_rnd_sigma_2['time'].values)]
    print(xa_sampling_rnd_sigma_2)
    times = xa_sampling_rnd_sigma_2.time.values
    n_times = len(times)
    xy_size = len(xa_sampling_rnd_sigma_2.lat.values) * len(xa_sampling_rnd_sigma_2.lon.values)
    cov_arr_shape = (n_times, xy_size)
    print('Creating err_cov with shape: ', cov_arr_shape)
    cov_arr = np.zeros(cov_arr_shape)
    #
    for emit_n in range(n_times):
        if (emit_n % 100) == 0:
            print('Processing ', emit_n, '/', n_times)
        uncorr_diag_vals = xa_sampling_rnd_sigma_2[emit_n].values.flatten()
        cov_arr[emit_n, :] = uncorr_diag_vals
    #
    print('Writing to ', out_error_covariance_npy)
    if savez_compressed:
        np.savez_compressed(out_error_covariance_npy, err_cov=cov_arr)
    else:
        np.save(out_error_covariance_npy, cov_arr, allow_pickle=False)
    #
    print('Task complete')


def sst_error_covariance_maker(yyyy,
                               mm,
                               savez_compressed=True):
    '''
    Create SST error covariance matrices
    Files are in year-month
    No seperation by ensemble member

    Diagonal terms should use "sigma2" values
    Do not use diagonal terms that appear in the other variable "cov"
    Use "cov" for off-diagonal terms
    '''
    #
    nc_path = '/noc/mpoc/surface_data/DCENTv1/'
    nc_path += 'DCENT_uncertainities/SST_Uncertainty_nc/5x5/'
    nc_file = nc_path + 'Uncertainty_reso_5_'+str(yyyy)+'_'+str(mm).zfill(2)+'.nc'
    xa_cubes = xa.open_dataset(nc_file)
    # ir_cubes = iris.load(nc_file)
    #
    # There should be a better way to do this, but for now we adhoc
    # load the original dcent file to check if a certain grid point could
    # possibly coastal, land or sea
    # Results are saved in terrain lookup tables
    clim_path = '/noc/mpoc/surface_data/DCENTv1/'
    clim_file = clim_path+'DCENT_monthly_climatology_1982_2014.nc'
    clim_var_sst = '1982--2014 Sea Surface Temperature Climatology.'
    clim_var_lsat = '1982--2014 Land Surface Air Temperature Climatology.'
    clim_cube_sst = iris.load_cube(clim_file, clim_var_sst)[mm-1]
    clim_cube_lsat = iris.load_cube(clim_file, clim_var_lsat)[mm-1]
    coastal_pixels = np.logical_and(~clim_cube_sst.data.mask, ~clim_cube_lsat.data.mask)
    land_pixels = np.logical_xor(coastal_pixels, ~clim_cube_lsat.data.mask)
    sea_pixels = np.logical_xor(coastal_pixels, ~clim_cube_sst.data.mask)
    # coast=2, land=1, sea=0
    cat_map = {0: 'sea', 1: 'land', 2: 'coast'}
    tac_map = {v: k for k, v in cat_map.items()}
    terrain_cat = coastal_pixels*tac_map['coast']+land_pixels*tac_map['land']+sea_pixels*tac_map['sea']
    terrain_cat_flatten = terrain_cat.flatten()
    terrain_cat_flatten_series = [cat_map[val] for val in terrain_cat_flatten.tolist()]
    terrain_cat_flatten_series = pd.Series(terrain_cat_flatten_series, dtype="category")
    #
    # Now into error covariances
    outpath = nc_path+'error_covariances/'
    out_latlon_table_feather = outpath+'latlon_'+str(yyyy)+'_'+str(mm).zfill(2)+'.feather'
    if savez_compressed:
        out_error_covariance_npy = outpath+'sst_error_covariance_'+str(yyyy)+'_'+str(mm).zfill(2)+'.npz'
    else:
        out_error_covariance_npy = outpath+'sst_error_covariance_'+str(yyyy)+'_'+str(mm).zfill(2)+'.npy'
    #
    # uncorr_samp_rnd_uncert_const = iris.AttributeConstraint(valid_max=lambda val: True)
    # corr_uncert_const = iris.AttributeConstraint(explanation2=lambda string: True)
    # ir_sampling_rnd_sigma_2 = ir_cubes.extract(uncorr_samp_rnd_uncert_const)[0] # Uncorrelated
    # ir_corr_cov = ir_cubes.extract(corr_uncert_const)[0] # Correlated
    # print(repr(ir_sampling_rnd_sigma_2))
    # print(repr(ir_corr_cov))
    #
    xa_sampling_rnd_sigma_2 = xa_cubes['sigma2']
    xa_corr_cov = xa_cubes['cov']
    lats = xa_sampling_rnd_sigma_2.lat.values
    lons = xa_sampling_rnd_sigma_2.lon.values
    xx, yy = np.meshgrid(lons, lats)
    xx_flatten = xx.flatten()
    yy_flatten = yy.flatten()
    idx = np.arange(len(xx_flatten))
    latlon_lookup_table = pd.DataFrame({'index': idx,
                                        'latitude': yy_flatten,
                                        'longitude': xx_flatten})
    latlon_lookup_table['terrain'] = terrain_cat_flatten_series
    print(latlon_lookup_table)
    print('Writing to ', out_latlon_table_feather)
    latlon_lookup_table.to_feather(out_latlon_table_feather)
    #
    uncorr_diag_vals = xa_sampling_rnd_sigma_2.values.flatten()
    # 20240916:
    # Duo Chan says use only these values for diagonal
    # Do not add diagonal values from the other variable
    cov_arr = np.diag(uncorr_diag_vals)
    #
    # Note: the actual idx follows MATLAB 1-indexing!
    # i, j >>> grid point i and grid point j
    idx_i, idx_lon_i, idx_lat_i = 0, 2, 3
    idx_j, idx_lon_j, idx_lat_j = 1, 4, 5
    nrows = xa_corr_cov.shape[1]
    #
    bad_cov_index_warnings = 0
    bad_sigma2_terrain_warnings = 0
    non_diag_rows = 0
    for row_idx in range(nrows):
        # Rember to subtract 1 from the actual index value!
        row_values = xa_corr_cov[:, row_idx].values
        cov_i = int(row_values[idx_i])-1
        cov_j = int(row_values[idx_j])-1
        #
        # 20240916: Per Duo Chan update, if cov_i == cov_j, continue
        if cov_i == cov_j:
            continue
        #
        nan_check = np.isnan(cov_arr[cov_i, cov_j])
        lat_i = yy_flatten[int(row_values[idx_i])-1]
        lon_i = xx_flatten[int(row_values[idx_i])-1]
        lat_j = yy_flatten[int(row_values[idx_j])-1]
        lon_j = xx_flatten[int(row_values[idx_j])-1]
        terrain_i = terrain_cat_flatten[cov_i]
        terrain_j = terrain_cat_flatten[cov_j]
        land_check_i = terrain_i == 1
        land_check_j = terrain_j == 1
        #
        # This is to check if nothing wrong in the processing the files
        # None of the rows should supposedely be flagged because
        # there are no land pixels within xa_corr_cov
        true_land = land_check_i or land_check_j
        valid_nan_check = nan_check and true_land
        if valid_nan_check:
            # Make sure there are no land pixels included...
            # 20240912 e.g. A coastal box near Brazil that is nearly all land (with tiny
            # amount of sea) can get flagged...
            old_val = cov_arr[cov_i, cov_j]
            assertion_msg = 'NAN in sigma2 and i and/or j point to land; this should not occur: '
            assertion_msg += '(i,j)=('+str(cov_i)+','+str(cov_j)+'); '
            assertion_msg += '(lat_i,lon_i,i_terrain)=('+str(lat_i)+','+str(lon_i)+','+cat_map[terrain_i]+'); '
            assertion_msg += '(lat_j,lon_j,j_terrain)=('+str(lat_j)+','+str(lon_j)+','+cat_map[terrain_j]+'); '
            assertion_msg += 'old_val = '+str(old_val)+'; bypassing.'
            # assert ~valid_nan_check, assertion_msg
            print(assertion_msg)
            warnings.warn(assertion_msg, UserWarning)
            bad_cov_index_warnings += 1
            # input("Press Enter to continue...")
            continue
        if nan_check:
            # Uncorrelated sigma**2 is NAN despite correlated terms suggest it should not
            old_val = cov_arr[cov_i, cov_j]
            assertion_msg = 'NAN in sigma2 but i and/or j point to sea or coastal; this should not occur: '
            assertion_msg += '(i,j)=('+str(cov_i)+','+str(cov_j)+'); '
            assertion_msg += '(lat_i,lon_i,i_terrain)=('+str(lat_i)+','+str(lon_i)+','+cat_map[terrain_i]+'); '
            assertion_msg += '(lat_j,lon_j,j_terrain)=('+str(lat_j)+','+str(lon_j)+','+cat_map[terrain_j]+'); '
            assertion_msg += 'old_val = '+str(old_val)+' ; forcing sigma2 to 0 .'
            # assert ~valid_nan_check, assertion_msg
            print(assertion_msg)
            warnings.warn(assertion_msg, UserWarning)
            cov_arr[cov_i, cov_j] = row_values[-1]
            bad_sigma2_terrain_warnings += 1
            # input("Press Enter to continue...")
        else:
            old_val = cov_arr[cov_i, cov_j]
            cov_arr[cov_i, cov_j] += row_values[-1]
        # Lat-Lon checks + progress report
        non_diag_rows += 1
        if (row_idx % 500) == 0:
            i_tuple = (yy_flatten[int(row_values[idx_i])-1],
                       xx_flatten[int(row_values[idx_i])-1],
                       lats[int(row_values[idx_lat_i])-1],
                       lons[int(row_values[idx_lon_i])-1])
            j_tuple = (yy_flatten[int(row_values[idx_j])-1],
                       xx_flatten[int(row_values[idx_j])-1],
                       lats[int(row_values[idx_lat_j])-1],
                       lons[int(row_values[idx_lon_j])-1])
            print(row_idx,
                  nrows,
                  i_tuple,
                  j_tuple,
                  old_val,
                  cov_arr[int(row_values[idx_i])-1, int(row_values[idx_j])-1])
    print('Number of normal rows processed = ', row_idx)
    print('Number of non-diagonal rows processed = ', non_diag_rows)
    print('bad_cov_index_warnings = ', bad_cov_index_warnings)
    print('bad_sigma2_terrain_warnings = ', bad_sigma2_terrain_warnings)
    #
    print('Writing to ', out_error_covariance_npy)
    if savez_compressed:
        np.savez_compressed(out_error_covariance_npy, err_cov=cov_arr)
    else:
        np.save(out_error_covariance_npy, cov_arr, allow_pickle=False)
    #
    print('Task complete')


def do_sst_err_cov():
    ''' process all dcent sst errcov into npz files '''
    yyyy0, yyyy1 = 1850, 2023
    yyyys = np.arange(yyyy0, yyyy1+1)
    for yyyy in yyyys:
        for mm in range(12):
            sst_error_covariance_maker(yyyy, mm+1)


def do_lsat_err_cov():
    ''' process all dcent lsat errcov into npz files '''
    ens0, ens1 = 1, 200
    enss = np.arange(ens0, ens1+1)
    for ens in enss:
        lsat_error_covariance_maker(ens)


def main():
    ''' MAIN '''
    print('--- MAIN ---')
    do_sst_err_cov()
    do_lsat_err_cov()


if __name__ == "__main__":
    main()
