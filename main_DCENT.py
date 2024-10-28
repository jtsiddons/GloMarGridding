################
# by A. Faulkner
# for python version 3.0 and up
################

#global
import sys
import os
import os.path
from datetime import datetime

#argument parser
import argparse
from configparser import ConfigParser
from collections import OrderedDict

#math tools 
import numpy as np

#data handling tools
from polars import date_range
import xarray as xr
import netCDF4 as nc

#self-written modules (from the same directory)
import observations as obs_module
import kriging as krig_module


class ConfigParserMultiValues(OrderedDict):
    def __setitem__(self, key, value):
        if key in self and isinstance(value, list):
            self[key].extend(value)
        else:
            super().__setitem__(key, value)

    @staticmethod
    def getlist(value):
        return value.splitlines()



def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("-config", dest="config", required=True, default="config.ini", help="INI file containing configuration settings", type=str)
    parser.add_argument("-year_start", dest="year_start", required=False, help="start year", type=int)
    parser.add_argument("-year_stop", dest="year_stop", required=False, help="end year", type=int)
    parser.add_argument("-month", dest="month", required=False, help="month", type=int)  # New Argument
    parser.add_argument("-member", dest="member", required=True, help="ensemble member: required argument", type = int, default = 0)
    parser.add_argument("-variable", dest="variable", required=False, help="variable to process: sst or lsat", type=str)
    parser.add_argument("-method", dest="method", default="simple", required=False, help="Kriging Method - one of \"simple\" or \"ordinary\"", type=str, choices=["simple", "ordinary"])
    parser.add_argument("-interpolation", dest="interpolation", default="ellipse", required=False, help="Interpolation covariance - one of \"distance\" or \"ellipse\"", type=str, choices=["distance", "ellipse"])
    args = parser.parse_args()
    
    config_file = args.config
    print(config_file)

    
    
    #load config options from ini file
    #this is done using an ini config file, which is located in the same direcotry as the python code
    #instantiate
    # ===== MODIFIED =====
    config = ConfigParser(strict=False,
                          empty_lines_in_values=False,
                          dict_type=ConfigParserMultiValues,
                          converters={"list": ConfigParserMultiValues.getlist})
    # ===== MODIFIED =====
    #parce existing config file
    config.read(config_file) #('config.ini' or 'three_step_kriging.ini')

    print(config)


    #read values from auxiliary_files section
    #for string use config.get
    #for boolean use config.getboolean
    #for int use config.getint
    #for float use config.getfloat
    #for list (multiple options for same key) use config.getlist
    

    #set boundaries for the domain
    lon_west  = config.getfloat('DCENT', 'lon_west') #-180. 
    lon_east  = config.getfloat('DCENT', 'lon_east') #180. 
    lat_south = config.getfloat('DCENT', 'lat_south') #-90.
    lat_north = config.getfloat('DCENT', 'lat_north') #90. 
    

    land_range = config.getfloat('DCENT', 'land_range') #1300
    land_sigma = config.getfloat('DCENT', 'land_sigma') #0.6
    land_matern = config.getfloat('DCENT', 'land_matern') #1.5
    
    sea_range =config.getfloat('DCENT', 'sea_range') #3100
    sea_sigma = config.getfloat('DCENT', 'sea_sigma') #1.2
    sea_matern = config.getfloat('DCENT', 'sea_matern') #1.5
    
    
    if args.year_start and args.year_stop:
        year_start = int(args.year_start)
        year_stop = int(args.year_stop)
    else:
        #start_date
        year_start = config.getint('DCENT', 'startyear')
        #end_date
        year_stop = config.getint('DCENT', 'endyear')
    print(year_start, year_stop)


    
    #location of the ICOADS observation files
    data_path = config.get('DCENT', 'observations')

    #location of landmasks
    landmask = config.get('DCENT', 'land_mask')

    #path to directory where the covariance(s) is/are located
    sst_error_cov_dir = config.get('sst', 'sst_error_covariance')
    lsat_error_cov_dir = config.get('lsat', 'lsat_error_covariance')

    #path to output directory
    output_directory = config.get('DCENT', 'output_dir')

    bnds = [lon_west, lon_east, lat_south, lat_north]
    #extract the latitude and longitude boundaries from user input
    lon_bnds, lat_bnds = (bnds[0], bnds[1]), (bnds[2], bnds[3])
    print(lon_bnds, lat_bnds)
    
    
    output_lat = np.arange(lat_bnds[0]+2.5, lat_bnds[-1]+2.5,5)
    output_lon = np.arange(lon_bnds[0]+2.5, lon_bnds[-1]+2.5,5)
    print(f'{output_lat =}')
    print(f'{output_lon =}')

    member = args.member

    #ts1 = datetime.now()
    #print(ts1)
    #read in observations for a chosen member
    obs = xr.open_dataset(
        os.path.join(
            data_path, 
            f'DCENT_ensemble_1850_2023_member_{member:03d}.nc'
        )
    )
    print('loaded observations')
    #ts2 = datetime.now()
    #print(ts2)
    print(obs)

    interpolation_covariance_type = str(args.interpolation)


    #what variable is being processed
    variable = args.variable or config.get("DCENT", "variable")
    match variable.lower():
        case 'sst':
            #read in sst error covariance
            error_cov = np.load(
                os.path.join(sst_error_cov_dir, 'sst_error_covariance_common.npz')
            )['err_cov']
            print('loaded sst error covariance')
            # #ts3 = datetime.now()
            #print(ts3)
            #(no of timesteps, no of gridboxes, no of gridboxes)
            #to extract what wanted chosen=[timestep,:,:]
            #replace NaNs with 0 before adding the matrices together
            if interpolation_covariance_type == 'distance':
                interpolation_covariance_path = config.get('sst', 'interpolation_covariance_seasig')
                interp_covariance = np.load(interpolation_covariance_path)
            elif interpolation_covariance_type == 'ellipse':
                interpolation_covariance_path = config.get('sst', 'ellipse_interpolation_covariance')

            var_range = sea_range
            var_sigma = sea_sigma
        
        case 'lsat':
            #ts0 = datetime.now()
            #print(ts0)
            #read in lsat error covariance for a chosen member
            error_cov = np.load(
                os.path.join(lsat_error_cov_dir,
                             f'lsat_error_covariance_{member}.npz')
            )['err_cov']
            print('loaded lsat error covariance')
            #(no of timesteps, no of gridboxeds 2592)
            #to extract what wanted chosen=[timestep,:] and then np.diag(chosen)

            if interpolation_covariance_type == 'distance':
                interpolation_covariance_path = config.get('lsat', 'interpolation_covariance_lndsig')
                interp_covariance = np.load(interpolation_covariance_path)
            elif interpolation_covariance_type == 'ellipse':
                interpolation_covariance_path = config.get('lsat', 'ellipse_interpolation_covariance')
    
            var_range = land_range
            var_sigma = land_sigma
        case _:
            raise ValueError(f"Unknown Variable {variable}")

    print(error_cov)
    print(error_cov.shape)
    print(len(error_cov.shape))
    

    output_directory = output_directory+f'/{variable}'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    print(output_directory)
    
    #create yearly output files
    year_list = range(int(year_start), int(year_stop)+1)
    for current_year in year_list:
        try:
            ncfile.close()  #make sure dataset is not already open.
        except (NameError, RuntimeError) as e:
            print(e)
            pass
        except Exception as e:  # Unknown Error
            raise e
            
        ncfilename = str(output_directory) 
        ncfilename = f"{current_year}_kriged"
        if member:
            ncfilename += f"_member_{member:03d}"
        ncfilename += ".nc"
        ncfilename = os.path.join(output_directory, ncfilename)
        
        ncfile = nc.Dataset(ncfilename,mode='w',format='NETCDF4_CLASSIC') 
        #print(ncfile)

        lat_dim = ncfile.createDimension('lat', len(output_lat))    # latitude axis
        lon_dim = ncfile.createDimension('lon', len(output_lon))    # longitude axis
        time_dim = ncfile.createDimension('time', None)         # unlimited axis
        
        # Define two variables with the same names as dimensions,
        # a conventional way to define "coordinate variables".
        lat = ncfile.createVariable('lat', np.float32, ('lat',))
        lat.units = 'degrees_north'
        lat.long_name = 'latitude'
        lon = ncfile.createVariable('lon', np.float32, ('lon',))
        lon.units = 'degrees_east'
        lon.long_name = 'longitude'
        time = ncfile.createVariable('time', np.float32, ('time',))
        time.units = f'days since {current_year}-01-15'
        time.long_name = 'time'
        #print(time)
                
        # Define a 3D variable to hold the data
        # note: unlimited dimension is leftmost
        krig_anom = ncfile.createVariable(f'{variable}_anomaly',
                                      np.float32,
                                      ('time','lat','lon'))
        krig_anom.standard_name = f"{variable} anomaly"
        krig_anom.units = 'deg C' # degrees Kelvin    

        # Define a 3D variable to hold the data
        krig_uncert = ncfile.createVariable(f'{variable}_anomaly_uncertainty',
                                      np.float32,
                                      ('time','lat','lon'))
        krig_uncert.units = 'deg C' # degrees Kelvin
        krig_uncert.standard_name = 'uncertainty' # this is a CF standard name
        
        # Define a 3D variable to hold the data
        grid_obs = ncfile.createVariable('observations_per_gridcell',np.float32,('time','lat','lon'))
        # note: unlimited dimension is leftmost
        grid_obs.units = '' # degrees Kelvin
        grid_obs.standard_name = 'Number of observations within each gridcell'

        
        # Write latitudes, longitudes.
        # Note: the ":" is necessary in these "write" statements
        lat[:] = output_lat #ds.lat.values
        lon[:] = output_lon #ds.lon.values


        for current_month in range(1, 13):
            timestep=current_month-1
            print('Current month and year: ', (current_month, current_year))

###############################################################################            
            mon_ds = obs.sel(time=np.logical_and(obs.time.dt.month == current_month, obs.time.dt.year == current_year))
            mon_df = mon_ds.to_dataframe().reset_index()
            #print(mon_df.columns)
            mon_df = mon_df.dropna(subset=[variable])
            mon_df.reset_index(inplace=True)
            print(mon_df)

            if interpolation_covariance_type == 'ellipse':
                interp_covariance = xr.open_dataset(interpolation_covariance_path + '/covariance_' + str(current_month).zfill(2) + '_v_eq_1p5_'+str(variable)+'_clipped.nc')['covariance'].values
                print(interp_covariance)
            
            
            #date_int = i * 12 + timestep
            date_int = (current_year - 1850) * 12 + timestep
            print(f'{current_year =}, {current_month =}')
            print(f'{date_int =}')
            match len(error_cov.shape):
                case 3:
                    error_covariance = error_cov[date_int,:,:]
                case 2:
                    error_covariance = np.diag(error_cov[date_int,:])
                case _:
                    raise ValueError("Error covariance.shape is not 2 or 3")

            #print(f'{error_covariance =}')
            ec_1 = error_covariance[~np.isnan(error_covariance)]
            ec_2 = ec_1[np.nonzero(ec_1)]
            #print('Non-nan and non-zero error covariance =', ec_2, len(ec_2))
            ec_idx = np.argwhere(np.logical_and(~np.isnan(error_covariance), error_covariance !=0.0))
            print('Index of non-nan and non-zero values =', ec_idx, len(ec_idx))
            
            #print(output_lat)
            #print(output_lon)
            mesh_lon, mesh_lat = np.meshgrid(output_lon, output_lat)
            #print(mesh_lat, mesh_lat.shape)
            #print(mesh_lon, mesh_lon.shape)
            #print(mon_ds[variable].values.squeeze().shape)
            print('-----------------')
            #since we're not using any landmask for this run
            #the line below:
            #cond_df, obs_flat_idx = obs_module.watermask_at_obs_locations(lon_bnds, lat_bnds, mon_df, mask_ds, mask_ds_lat, mask_ds_lon)
            #mon_flat_idx = cond_df['flattened_idx'][:]
            #can be substituted with:
            lat_idx, grid_lat = obs_module.find_nearest(output_lat, mon_df.lat)
            lon_idx, grid_lon = obs_module.find_nearest(output_lon, mon_df.lon)

            mon_df['grid_lat'] = grid_lat
            mon_df['grid_lon'] = grid_lon
            mon_df['lat_idx'] = lat_idx
            mon_df['lon_idx'] = lon_idx
            
            idx_tuple = np.array([lat_idx, lon_idx]) #list(zip(lat_idx, lon_idx))
            #print(f'{idx_tuple =}')
            mon_flat_idx = np.ravel_multi_index(idx_tuple, mesh_lat.shape, order='C') #row-major
            #print(f'{mon_flat_idx =}') #it's the same as ec_idx
            #print(f'{sorted(mon_flat_idx) =}')
            
            #diff = sorted(mon_flat_idx) - np.unique(mon_flat_idx)
            # diff results in 0s as unique idx is the same as sorted mon_flat_idx
            
            mon_df['gridbox'] = mon_flat_idx
            #print(mon_df)
            mon_df['error_covariance_diagonal'] = error_covariance[mon_flat_idx,mon_flat_idx]
            print(mon_df)
            mon_df = mon_df.dropna(subset=['error_covariance_diagonal'])
            mon_df.reset_index(inplace=True)
            print(mon_df)
            
            #count obs per grid for output
            gridbox_counts = mon_df['gridbox'].value_counts()
            gridbox_count_np = gridbox_counts.to_numpy()
            gridbox_id_np = gridbox_counts.index.to_numpy()
            del gridbox_counts
            water_mask = np.copy(mesh_lat)
            grid_obs_2d = krig_module.result_reshape_2d(gridbox_count_np, gridbox_id_np, water_mask)

            
            #need to either add weights (which will be just 1 everywhere as obs are gridded)
            #krige obs onto gridded field
            _, W = obs_module.dist_weight(mon_df, dist_fn=obs_module.haversine_gaussian, R=6371.0, r=var_range, s=var_sigma)
            #print(W)

            
            grid_idx = np.array(sorted(mon_df['gridbox'])) #sorted?
            #print(error_covariance, error_covariance.shape)
            error_covariance = error_covariance[grid_idx[:,None],grid_idx[None,:]]
            #print(np.argwhere(np.isnan(np.diag(error_covariance))))
            #print(f'{error_covariance =}, {error_covariance.shape =}')
            
            
            anom, uncert = krig_module.kriging_simplified(grid_idx, W, np.asarray(mon_df[variable].values), interp_covariance, error_covariance, method=args.method)
            print('Kriging done, saving output')
            print(anom)
            print(uncert)
            print(grid_obs_2d)

            
            #reshape output into 2D
            anom = np.reshape(anom, mesh_lat.shape)
            uncert = np.reshape(uncert, mesh_lat.shape)
            
            # Write the data.  
            #This writes each time slice to the netCDF
            krig_anom[timestep,:,:] = anom #ordinary_kriging
            krig_uncert[timestep,:,:] = uncert #ordinary_kriging
            grid_obs[timestep,:,:] = grid_obs_2d.astype(np.float32)
            print("-- Wrote data")
            print(timestep, current_month)
            
        # Write time
        times = (
            date_range(start=datetime(current_year, 1, 15), end=datetime(current_year, 12, 15), interval="1mo", eager=True)
            - datetime(current_year, 1, 15)
        ).dt.total_days().to_numpy()
        print(f"{times = }")

        time[:] = times

        print(time)    
        # first print the Dataset object to see what we've got
        print(ncfile)
        # close the Dataset.
        ncfile.close()
        print('Dataset is closed!')
        

if __name__ == '__main__':
    main(sys.argv[1:])
