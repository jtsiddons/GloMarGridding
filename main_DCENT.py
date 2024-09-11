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
    parser.add_argument("-month", dest="month", required=True, help="month")  # New Argument
    parser.add_argument("-height_member", dest="height_member", required=False, help="height member: if height member is 0, no height adjustment is performed.", type = int, default = 0)
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
    land_matern = config.getfloat('DCENT', 'land_,matern') #1.5
    
    sea_range =config.getfloat('DCENT', 'sea_range') #31300
    sea_sigma = config.getfloat('DCENT', 'sea_sigma') #1.2
    sea_matern = config.getfloat('DCENT', 'sea_matern') #1.5
    
    """
    member = args.height_member
    print(member)
    if member > 0:
        height_adjustment_path = config.get('DCENT', 'height_adjustments')
        adjusted_height = config.getint('DCENT', 'adjusted_height')
    """
    
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
    landmask = config.getfloat('DCENT', 'land_mask')

    #location of climatology 
    climatology = config.get('DCENT', 'climatology')

    #path to directory where the covariance(s) is/are located
    error_cov_dir = config.get('DCENT', 'error_covariance')

    #path to output directory
    output_directory = config.get('DCENT', 'output_dir')

    
    
    bnds = [lon_west, lon_east, lat_south, lat_north]
    #extract the latitude and longitude boundaries from user input
    lon_bnds, lat_bnds = (bnds[0], bnds[1]), (bnds[2], bnds[3])
    print(lon_bnds, lat_bnds)
    
    output_lat = np.arange(lat_bnds[0]+0.5, lat_bnds[-1]+0.5,1)
    output_lon = np.arange(lon_bnds[0]+0.5, lon_bnds[-1]+0.5,1)
    print(output_lat)
    print(output_lon)
    
    """
    #land-water-mask for observations
    water_mask_dir = config.get('MAT', 'covariance')
    mask_ds, mask_ds_lat, mask_ds_lon = cov_module.get_landmask(water_mask_dir, month=1)
    """
    
    """
    water_mask_file = config.getlist('landmask', 'land_mask')
    print(water_mask_file)
    #for ellipse Atlatic 0, for ellipse world 1, for ESA world 2
    water_mask_file = water_mask_file[1]
    print(water_mask_file)
    
    mask_ds, mask_ds_lat, mask_ds_lon = obs_module.landmask(water_mask_file, lat_south,lat_north, lon_west,lon_east)
    print('----')
    print(mask_ds)
    """
    
    
    climatology = obs_module.read_climatology(climatology, lat_north,lat_south, lon_west,lon_east)
    print(climatology)
    clim = climatology.t10m_clim_day
    print(clim)
    del climatology
    #while doing pentad processing, this will set "mid-pentads" dates for the year
    
    
    year_list = list(range(int(year_start), int(year_stop)+1,1))
    # ===== MODIFIED =====
    month_list = list(range(mm2process, mm2process+1, 1))
    # ===== MODIFIED =====

    for i in range(len(year_list)):
        current_year = year_list[i]

        #add MetOffice pentads here
        yr_rng = pd.date_range('1970/01/03','1970/12/31',freq='5D')

        # ===== NEW =====
        yr_rng = yr_rng[yr_rng.month == mm2process]
        # ===== NEW =====

        times2 = [j.replace(year=current_year) for j in yr_rng]
        print(times2)
        times_series = pd.Series(times2)
        by_month = list(times_series.groupby(times_series.map(lambda x: x.month)))
        print(by_month)

        try:
            ncfile.close()  #make sure dataset is not already open.
        except: 
            pass
            
        ncfilename = str(output_directory) 
        ncfilename = f"{current_year}_{mm2process:02d}_kriged_MAT"
        if member:
            ncfilename += f"_heightmember_{member:03d}"
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
        time.units = 'days since %s-01-01' % (str(current_year))
        time.long_name = 'time'
        #print(time)
        
        # Define a 3D variable to hold the data
        if member:      
            krig_anom = ncfile.createVariable(f"{variable}_anomaly_{adjusted_height}m",
                                      np.float32, ('time','lat','lon'))
            # note: unlimited dimension is leftmost
            krig_anom.standard_name = f"{variable} anomaly at {adjusted_height} m"
            krig_anom.height = str(adjusted_height)
            krig_anom.ensemble_member = member
        else:
            krig_anom = ncfile.createVariable("mat_anomaly", np.float32, ('time', 'lat', 'lon'))
            krig_anom.standard_name = "MAT anomaly"
        krig_anom.units = 'deg C' # degrees Kelvin    

        # Define a 3D variable to hold the data
        krig_uncert = ncfile.createVariable('mat_anomaly_uncertainty',
                                      np.float32,
                                      ('time','lat','lon')) # note: unlimited dimension is leftmost
        krig_uncert.units = 'deg C' # degrees Kelvin
        krig_uncert.standard_name = 'uncertainty' # this is a CF standard name
        
        # Write latitudes, longitudes.
        # Note: the ":" is necessary in these "write" statements
        lat[:] = output_lat #ds.lat.values
        lon[:] = output_lon #ds.lon.values
        
        for timestep in range(len(month_list)):

            current_month = month_list[timestep]
            #print(current_month) 
            mon_df = obs_qc_module.main(data_path, year=current_year, month=current_month)
            
            print('Current month and year: ', (current_month, current_year))
            
            # calculate covariance based on the observations for given year and month
            covariance = cov_module.calculate_covariance()
            print(covariance)

            #read in landmask corresponding to the covariance
            mask_ds, mask_ds_lat, mask_ds_lon = cov_module.get_landmask()
            print(mask_ds)
            """
            # NOT NEEDED AT CURRENT STAGE - MIGHT BE NEEDED IN THE FUTURE
            # read in observations
            obs_df = obs_qc_module.MAT_main(data_path, qc_path, qc_path_2, qc_mat, year=current_year, month=current_month)
            print(obs_df)
            day_night = pl.from_pandas(obs_df[['uid', 'datetime', 'lon', 'lat']]) # required cols for is_daytime
            day_night = day_night.pipe(is_daytime)
            obs_df = obs_df.merge(day_night.select(['uid', 'is_daytime']).to_pandas(), on='uid')
            del day_night

            #filter day (1) or night(0)
            obs_df = obs_df[obs_df['is_daytime'] == 0]
            print(obs_df) #[['local_datetime', 'is_daytime']])
            """
            
            
            current_date = datetime(current_year,current_month,15)
            print('----------')
            print('timestep', timestep)
            print('----------')
            print(current_date)
            
            esa_climatology = climatology['climatology']
            print(esa_climatology)
            
            #add climatology value and calculate the SST anomaly
            #mon_df = obs_module.extract_clim_anom(esa_climatology, mon_df)
            mon_df = obs_qc_module.SST_match_climatology_to_obs(climatology, mon_df)
 
            cond_df, obs_flat_idx = obs_module.watermask_at_obs_locations(lon_bnds, lat_bnds, mon_df, mask_ds, mask_ds_lat, mask_ds_lon)
                
            mon_flat_idx = cond_df['flattened_idx'][:]

            #read in error covariance for a given month and year
            #should be the same size as the total number of observations for that timestep
            obs_covariance, W = obs_module.red_error_covariance()
            #print(obs_covariance)
            #print(W)
            
            #krige obs onto gridded field
            anom, uncert = krig_module.kriging_main(covariance, mask_ds, cond_df, mon_flat_idx, obs_covariance, W, kriging_method=args.method)
            print('Kriging done, saving output')

            # Write the data.  
            #This writes each time slice to the netCDF
            krig_anom[timestep,:,:] = anom #ordinary_kriging
            krig_uncert[timestep,:,:] = uncert #ordinary_kriging
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
