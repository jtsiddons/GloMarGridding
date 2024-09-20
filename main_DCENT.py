################
# by A. Faulkner
# for python version 3.0 and up
################

#global
import sys, os, re
import glob
import os.path
from os import path
from os.path import isfile, join

#argument parser
import argparse
try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser  # ver. < 3.0                                               
from collections import OrderedDict
from configparser import ConfigParser  

#math tools 
import numpy as np
import math
from scipy.linalg import block_diag

#plotting tools
import matplotlib.pyplot as plt

#timing tools
import timeit
from calendar import isleap
from calendar import monthrange
#import datetime as dt
from datetime import datetime, timedelta
from netCDF4 import date2num, num2date

#data handling tools
import pandas as pd
import xarray as xr
import netCDF4 as nc
from functools import partial
import polars as pl

#self-written modules (from the same directory)
import covariance_calculation as cov_cal
import covariance as cov_module
import observations as obs_module
import observations_plus_qc as obs_qc_module
import kriging as krig_module


####
#for plotting
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import simple_plots as cp


#PyCOADS functions
from PyCOADS.utils.solar import sun_position, is_daytime




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
    parser.add_argument("-member", dest="member", required=True, help="ensemble member: required argument", type = int, default = 0)
    parser.add_argument("-variable", dest="variable", required=False, help="variable to process: sst or lsat")
    parser.add_argument("-method", dest="method", default="simple", required=False, help="Kriging Method - one of \"simple\" or \"ordinary\"")
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
    
    sea_range =config.getfloat('DCENT', 'sea_range') #31300
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
    sst_error_cov_dir = config.get('DCENT', 'sst_error_covariance')
    lsat_error_cov_dir = config.get('DCENT', 'lsat_error_covariance')

    #path to where the global interpolation covariance is
    sea_interp_cov = config.get('DCENT', 'interpolation_covariance_seasig')
    lnd_interp_cov = config.get('DCENT',  'interpolation_covariance_lndsig')

    #path to output directory
    output_directory = config.get('DCENT', 'output_dir')

    #what variable is being processed
    if args.variable:
        variable = str(args.variable)
    else:
        variable = config.get('DCENT', 'variable')
    
    
    bnds = [lon_west, lon_east, lat_south, lat_north]
    #extract the latitude and longitude boundaries from user input
    lon_bnds, lat_bnds = (bnds[0], bnds[1]), (bnds[2], bnds[3])
    print(lon_bnds, lat_bnds)
    
    output_lat = np.arange(lat_bnds[0]+2.5, lat_bnds[-1]+2.5,5)
    output_lon = np.arange(lon_bnds[0]+2.5, lon_bnds[-1]+2.5,5)
    print(output_lat)
    print(output_lon)

    

    
    member = int(args.member)

    #ts1 = datetime.now()
    #print(ts1)
    #read in observations for a chosen member
    obs = xr.open_dataset(data_path+'/DCENT_ensemble_1850_2023_member_'+str(member).zfill(3)+'.nc')
    print('loaded observations')
    #ts2 = datetime.now()
    #print(ts2)
    print(obs)



    
    if variable == 'sst':
        #read in sst error covariance
        error_cov = np.load(sst_error_cov_dir+'/sst_error_covariance_common.npz')['err_cov']
        print('loaded sst error covariance')
        #ts3 = datetime.now()
        #print(ts3)
        #(no of timesteps, no of gridboxes, no of gridboxes)
        #to extract what wanted chosen=[timestep,:,:]
        #replace NaNs with 0 before adding the matrices together
        print(error_cov)
        print(error_cov.shape)
        print(len(error_cov.shape))

        interp_covariance = np.load(sea_interp_cov)

        
    elif variable == 'lsat':
        #ts0 = datetime.now()
        #print(ts0)
        #read in lsat error covariance for a chosen member
        error_cov = np.load(lsat_error_cov_dir+'/lsat_error_covariance_'+str(member)+'.npz')['err_cov']
        print('loaded lsat error covariance')
        print(error_cov)
        print(error_cov.shape)
        print(len(error_cov.shape))
        #(no of timesteps, no of gridboxeds 2592)
        #to extract what wanted chosen=[timestep,:] and then np.diag(chosen)

        interp_covariance = np.load(lnd_interp_cov)

        
    
    #create yearly output files
    year_list = list(range(int(year_start), int(year_stop)+1,1))

    for i in range(len(year_list)):
        current_year = year_list[i]
        
        try:
            ncfile.close()  #make sure dataset is not already open.
        except: 
            pass
            
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
        for timestep in range(len(month_list)):

            current_month = month_list[timestep]
            print('Current month and year: ', (current_month, current_year))

###############################################################################            
            mon_ds = obs.sel(time=np.logical_and(obs.time.dt.month == current_month, obs.time.dt.year == current_year))
            print(mon_ds)
            mon_df = mon_ds.to_dataframe().reset_index()
            print(mon_df)
            print(mon_df.columns)
            mon_df = mon_df.dropna(subset=[variable])
            mon_df.reset_index(inplace=True)
            print(mon_df)

            
            
            date_int = i * 12 + timestep
            print(f'{i =}, {current_year =}, {timestep =}, {current_month =}')
            print(f'{date_int =}')
            if len(error_cov.shape) == 3:
                error_covariance = error_cov[date_int,:,:]
            elif len(error_cov.shape) == 2:
                error_covariance = np.diag(error_cov[date_int,:])
            print(error_covariance)
            
            # add interpolation (distance-based) and error covariances (lsat and sst) for given year and month
            joined_covariance = interp_covariance + error_covariance
            print(joined_covariance)
            
            """
            current_date = datetime(current_year,current_month,15)
            print('----------')
            print('timestep', timestep)
            print('----------')
            print(current_date)
            """
            print(output_lat)
            print(output_lon)
            mesh_lon, mesh_lat = np.meshgrid(output_lon, output_lat)
            print(mesh_lat, mesh_lat.shape)
            print(mesh_lon, mesh_lon.shape)
            print(mon_ds[variable].values.squeeze().shape)
            print('-----------------')
            #since we're not using any landmask for this run
            #the line below:
            #cond_df, obs_flat_idx = obs_module.watermask_at_obs_locations(lon_bnds, lat_bnds, mon_df, mask_ds, mask_ds_lat, mask_ds_lon)
            #mon_flat_idx = cond_df['flattened_idx'][:]
            #can be substituted with:
            lat_idx, grid_lat = obs_module.find_nearest(output_lat, mon_df.lat)
            lon_idx, grid_lon = obs_module.find_nearest(output_lon, mon_df.lon)
            idx_tuple = np.array([lat_idx, lon_idx])
            print(f'{idx_tuple =}')
            mon_flat_idx = np.ravel_multi_index(idx_tuple, mesh_lat.shape, order='C') #row-major
            print(mon_flat_idx)

            #count obs per grid for output
            mon_df["gridbox"] = mon_flat_idx
            gridbox_counts = mon_df['gridbox'].value_counts()
            gridbox_count_np = gridbox_counts.to_numpy()
            gridbox_id_np = gridbox_counts.index.to_numpy()
            del gridbox_counts
            water_mask = np.copy(mesh_lat)
            grid_obs_2d = krig_module.result_reshape_2d(gridbox_count_np, gridbox_id_np, water_mask)
            STOP
            #need to either add weights (which will be just 1 everywhere as obs are gridded)
            #or ammend the code to skip weights
            #krige obs onto gridded field
            anom, uncert = krig_module.kriging_main(interp_covariance, mask_ds, cond_df, mon_flat_idx, error_covariance, W, kriging_method=args.method)
            print('Kriging done, saving output')

            # Write the data.  
            #This writes each time slice to the netCDF
            krig_anom[timestep,:,:] = anom #ordinary_kriging
            krig_uncert[timestep,:,:] = uncert #ordinary_kriging
            grid_obs[timestep,:,:] = grid_obs_2d.astype(np.float32)
            print("-- Wrote data")
            print(timestep, current_date)
            
        # Write time
        #pd.date_range takes month/day/year as input dates
        clim_times_updated = [j.replace(year=current_year) for j in pd.to_datetime(clim_times.data)]
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
