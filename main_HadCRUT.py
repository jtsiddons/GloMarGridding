################
# by A. Faulkner
# for python version 3.0 and up
################

#global
import sys, os
import os.path

#argument parser
import argparse
from configparser import ConfigParser                                             
from collections import OrderedDict
from configparser import ConfigParser  

#math tools 
import numpy as np

#plotting tools

#import datetime as dt
from netCDF4 import date2num

#data handling tools
import pandas as pd
import xarray as xr
import netCDF4 as nc
import h5netcdf
from zipfile import ZipFile

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
    parser.add_argument("-config", dest="config", required=False, default="config.ini", help="INI file containing configuration settings")
    parser.add_argument("-year_start", dest="year_start", required=False, help="start year")
    parser.add_argument("-year_stop", dest="year_stop", required=False, help="end year")
    parser.add_argument("-month", dest="month", required=False, help="month")  # New Argument
    parser.add_argument("-member", dest="member", required=False, help="ensemble member: required argument", type = int, default = 0)
    parser.add_argument("-variable", dest="variable", required=True, help="variable to process: sst or lsat")
    parser.add_argument("-method", dest="method", default="ordinary", required=False, help="Kriging Method - one of \"simple\" or \"ordinary\"")
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
    config.read(config_file) 
    print(config)


    #read values from auxiliary_files section
    #for string use config.get
    #for boolean use config.getboolean
    #for int use config.getint
    #for float use config.getfloat
    #for list (multiple options for same key) use config.getlist
    

    #set boundaries for the domain
    lon_west  = config.getfloat('HadCRUT', 'lon_west') #-180. 
    lon_east  = config.getfloat('HadCRUT', 'lon_east') #180. 
    lat_south = config.getfloat('HadCRUT', 'lat_south') #-90.
    lat_north = config.getfloat('HadCRUT', 'lat_north') #90. 
    

    
    
    if args.year_start and args.year_stop:
        year_start = int(args.year_start)
        year_stop = int(args.year_stop)
    else:
        #start_date
        year_start = config.getint('HadCRUT', 'startyear')
        #end_date
        year_stop = config.getint('HadCRUT', 'endyear')
    print(year_start, year_stop)


    
    #path to output directory
    output_directory = config.get('HadCRUT', 'output_dir')


    #read in member from comand line
    member = int(args.member)


    #what variable is being processed
    if args.variable:
        variable = str(args.variable)
    else:
        print('Please supply from the command line the variable to process')
    
    bnds = [lon_west, lon_east, lat_south, lat_north]
    #extract the latitude and longitude boundaries from user input
    lon_bnds, lat_bnds = (bnds[0], bnds[1]), (bnds[2], bnds[3])
    print(lon_bnds, lat_bnds)
    
    
    output_lat = np.arange(lat_bnds[0]+2.5, lat_bnds[-1]+2.5,5)
    output_lon = np.arange(lon_bnds[0]+2.5, lon_bnds[-1]+2.5,5)
    print(f'{output_lat =}')
    print(f'{output_lon =}')
    
    if variable == 'tos':
        data_path = config.get('sst', 'observations')
        error_covariance_path = config.get('sst', 'error_covariance')
        sampling_uncertainty = config.get('sst', 'sampling_uncertainty')
        uncorrelated_uncertainty = config.get('sst', 'uncorrelated_uncertainty')
        interpolation_covariance_path = config.get('sst', 'interpolation_covariance')
        var_range =config.getfloat('HadCRUT', 'sea_range') #3100
        var_sigma = config.getfloat('HadCRUT', 'sea_sigma') #1.2
        var_matern = config.getfloat('HadCRUT', 'sea_matern') #1.5

        print(f'Processing for variable {variable}')
        obs = xr.open_dataset(data_path) #+'/HadCRUT_ensemble_1850_2023_member_'+str(member).zfill(3)+'.nc')
        print('loaded observations')
        print(obs)

        #error_cov = xr.open_dataset(error_covariance)
        #print('loaded error covariance')
        #print(error_cov)
        sampling = xr.open_dataset(sampling_uncertainty)
        uncorrelated = xr.open_dataset(uncorrelated_uncertainty)
        print('loaded uncertainties')
        print(sampling)
        print(uncorrelated)
        
        interp_covariance = np.load(interpolation_covariance_path)
        print('loaded interpolation covariance')
        print(interp_covariance)


        #create yearly output files
        year_list = list(range(int(year_start), int(year_stop)+1,1))

        for current_year in year_list:
        
            try:
                ncfile.close()  #make sure dataset is not already open.
            except:     
                pass
            
            ncfilename = str(output_directory) 
            ncfilename = f"{current_year}_kriged"
            if member:
                ncfilename += f"_member_{member:03d}"
            ncfilename += ".nc"
            #ncfilename = os.path.join(output_directory+f'/{variable}', ncfilename)
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
            time.units = 'days since %s-01-15' % (str(current_year))
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


            month_list = list(range(1,13,1))
            for current_month in month_list:
                timestep=current_month-1
                print('Current month and year: ', (current_month, current_year))

#################            
                mon_ds = obs.sel(time=np.logical_and(obs.time.dt.month == current_month, obs.time.dt.year == current_year))
                mon_df = mon_ds.to_dataframe().reset_index()
                print(mon_df.columns)
                mon_df = mon_df.drop_duplicates(subset=['latitude', 'longitude', 'tos', 'time'])
                mon_df = mon_df.dropna(subset=[variable])
                mon_df.reset_index(inplace=True)

                mon_df = mon_df.rename(columns={'latitude':'lat', 'longitude':'lon'})

                print(mon_df)

                #FOR TOS from HadSST4:
                error_cov = xr.open_dataset(error_covariance_path+'/HadSST.4.0.1.0_error_covariance_'+str(current_year)+str(current_month).zfill(2)+'.nc')['tos_cov'].values
                print(error_cov)
                print(error_cov.shape)
                sampling_mon = sampling.sel(time=np.logical_and(sampling.time.dt.month == current_month, sampling.time.dt.year == current_year))['tos_unc'].values
                uncorrelated_mon = uncorrelated.sel(time=np.logical_and(uncorrelated.time.dt.month == current_month, uncorrelated.time.dt.year == current_year))['tos_unc'].values
                print(sampling_mon)
                print(sampling_mon.shape)
                print(uncorrelated_mon)
                print(uncorrelated_mon.shape)
                joined_mon = sampling_mon * sampling_mon + uncorrelated_mon * uncorrelated_mon
                print(joined_mon)
                print(joined_mon.shape)
                #joined = np.power(joined_mon, 0.5)
                #print(joined)
                unc_1d = np.reshape (joined_mon,(2592 ,1))
                print(unc_1d)
                #covariance2 = np.diag (np.reshape (unc_1d * unc_1d ,(2592)))
                covariance2 = np.diag (np.reshape (unc_1d ,(2592)))
                print(covariance2)
                error_covariance = error_cov + covariance2
                del error_cov
                
                print(error_covariance)
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
                print(mon_df)
                
                idx_tuple = np.array([lat_idx, lon_idx])
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
                
                
                anom, uncert = krig_module.kriging_simplified(grid_idx, W, mon_df[variable].values, interp_covariance, error_covariance, method=args.method)
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
            clim_times = pd.date_range(start='01/15/2000', end='12/15/2000', periods=12)
            #pd.date_range takes month/day/year as input dates
            clim_times_updated = [j.replace(year=current_year, day=15) for j in pd.to_datetime(clim_times)]
            print(clim_times_updated)
            dates_ = pd.Series(clim_times_updated)
            dates = dates_.dt.to_pydatetime() # Here it becomes date
            print('pydate', dates)
            times = date2num(dates, time.units)
            print(times)

            time[:] = times

            print(time)    
            # first print the Dataset object to see what we've got
            print(ncfile)
            # close the Dataset.
            ncfile.close()
            print('Dataset is closed!')




##########################################################################################################
        
    elif variable == 'tas':
        data_path = config.get('lsat', 'observations')
        error_covariance_path = config.get('lsat', 'error_covariance')
        interpolation_covariance_path = config.get('lsat', 'interpolation_covariance')
        var_range = config.getfloat('HadCRUT', 'land_range') #1300
        var_sigma = config.getfloat('HadCRUT', 'land_sigma') #0.6
        var_matern = config.getfloat('HadCRUT', 'land_matern') #1.5
    
        print(f'Processing for variable {variable}')
        obs = xr.open_dataset(data_path) #+'/HadCRUT_ensemble_1850_2023_member_'+str(member).zfill(3)+'.nc')
        print('loaded observations')
        print(obs)


        error_meas = np.load(error_covariance_path +'/CRUTEM.5.0.2.0.measurement_sampling.npz')['err_cov']
        error_stat = np.load(error_covariance_path +'/CRUTEM.5.0.2.0.station_uncertainty.npz')['err_cov']
        print('loaded error covariance')
        print(error_meas)
        print(error_stat)
        error_cov = error_meas + error_stat
        
        interp_covariance = np.load(interpolation_covariance_path)
        print('loaded interpolation covariance')
        print(interp_covariance)

    
        #create yearly output files
        year_list = list(range(int(year_start), int(year_stop)+1,1))

        for current_year in year_list:
            
            try:
                ncfile.close()  #make sure dataset is not already open.
            except: 
                pass
                
            ncfilename = str(output_directory) 
            ncfilename = f"{current_year}_kriged"
            if member:
                ncfilename += f"_member_{member:03d}"
            ncfilename += ".nc"
            #ncfilename = os.path.join(output_directory+f'/{variable}', ncfilename)
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
            time.units = 'days since %s-01-15' % (str(current_year))
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


            
            month_list = list(range(1,13,1))
            for current_month in month_list:
                timestep=current_month-1
                print('Current month and year: ', (current_month, current_year))

    ################            
                mon_ds = obs.sel(time=np.logical_and(obs.time.dt.month == current_month, obs.time.dt.year == current_year))
                mon_df = mon_ds.to_dataframe().reset_index()
                print(mon_df.columns)
                mon_df = mon_df.drop_duplicates(subset=['latitude', 'longitude', 'tas', 'time'])
                mon_df = mon_df.dropna(subset=[variable])
                mon_df.reset_index(inplace=True)

                mon_df = mon_df.rename(columns={'latitude':'lat', 'longitude':'lon'})

                print(mon_df)

                # FOR TAS from CRUTEM:
                #date_int = i * 12 + timestep
                date_int = (current_year - 1850) * 12 + timestep
                print(f'{current_year =}, {current_month =}')
                print(f'{date_int =}')
                if len(error_cov.shape) == 3:
                    error_covariance = error_cov[date_int,:,:]
                elif len(error_cov.shape) == 2:
                    error_covariance = np.diag(error_cov[date_int,:])
                print(f'{error_covariance =}')
                ec_1 = error_covariance[~np.isnan(error_covariance)]
                ec_2 = ec_1[np.nonzero(ec_1)]
                print('Non-nan and non-zero error covariance =', ec_2, len(ec_2))
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
                
                idx_tuple = np.array([lat_idx, lon_idx])
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
                
                
                anom, uncert = krig_module.kriging_simplified(grid_idx, W, mon_df[variable].values, interp_covariance, error_covariance, method=args.method)
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
            clim_times = pd.date_range(start='01/15/2000', end='12/15/2000', periods=12)
            #pd.date_range takes month/day/year as input dates
            clim_times_updated = [j.replace(year=current_year, day=15) for j in pd.to_datetime(clim_times)]
            print(clim_times_updated)
            dates_ = pd.Series(clim_times_updated)
            dates = dates_.dt.to_pydatetime() # Here it becomes date
            print('pydate', dates)
            times = date2num(dates, time.units)
            print(times)

            time[:] = times

            print(time)    
            # first print the Dataset object to see what we've got
            print(ncfile)
            # close the Dataset.
            ncfile.close()
            print('Dataset is closed!')
        





if __name__ == '__main__':
    main(sys.argv[1:])
