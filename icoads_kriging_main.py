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

# IMPORTANT: Environmental Variables to limit Numpy
os.environ["OMP_NUM_THREADS"] = '16'
os.environ["OPENBLAS_NUM_THREADS"] = '16'
os.environ["MKL_NUM_THREADS"] = '16'
os.environ["VECLIB_MAXIMUM_THREADS"] = '16'
os.environ["NUMEXPR_NUM_THREADS"] = '16'

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
from datetime import datetime
from netCDF4 import date2num, num2date

#data handling tools
import pandas as pd
import xarray as xr
import netCDF4 as nc
from functools import partial

#self-written modules (from the same directory)
import covariance_calculation as cov_cal
import covariance as cov_module
import observations as obs_module
import observations_plus_qc as obs_qc_module
import kriging as krig_module
import output as out_module


####
#for plotting
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import simple_plots as cp







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
    args = parser.parse_args()
    
    config_file = args.config
    print(config_file)

    
    
    #load config options from ini file
    #this is done using an ini config file, which is located in the same direcotry as the python code
    #instantiate
    config = ConfigParser(strict=False, empty_lines_in_values=False, dict_type=ConfigParserMultiValues, converters={"list": ConfigParserMultiValues.getlist})
    #parce existing config file
    config.read(config_file) #('config.ini' or 'three_step_kriging.ini')
    
    print(config)
    
        
    #read values from auxiliary_files section
    #for string use config.get
    #for boolean use config.getboolean
    #for int use config.getint
    #for float use config.getfloat
    #for list (multiple options for same key) use config.getlist
    
    #location of ESA CCI files for the covariance creation and output grid
    cci_directory = config.get('observations', 'esa_cci')
    cci_climatology = config.get('observations', 'esa_climatology')

    #what are we processing - variable in the files from cci_directory
    ds_varname = config.getlist('variable_name', 'variable')
    #for sst 0, for sst_anomaly 1
    ds_varname = ds_varname[1]
    
    #set boundaries for the domain
    lon_west  = config.getfloat('domain', 'lon_west') #-180. 
    lon_east  = config.getfloat('domain', 'lon_east') #180. 
    lat_south = config.getfloat('domain', 'lat_south') #-90.
    lat_north = config.getfloat('domain', 'lat_north') #90. 
    
    #set step size (grid size)
    dlon = config.getfloat('grid', 'dlon') #0.25
    dlat = config.getfloat('grid', 'dlat') #0.25
    
    #location of the ICOADS observation files
    data_path = config.get('observations', 'observations')
    #location og QC flags in GROUPS subdirectories
    qc_path = config.get('observations', 'qc_flags_joe')
    
    
    if args.year_start and args.year_stop:
        year_start = int(args.year_start)
        year_stop = int(args.year_stop)
    else:
        #start_date
        year_start = config.getint('time_period', 'startyear')
        #end_date
        year_stop = config.getint('time_period', 'endyear')
    print(year_start, year_stop)
    
    #path where the covariance(s) is/are located
    #if single covariance, then full path
    #if several different covariances, then path to directory
    cov_dir = config.get('covariance', 'covariance_path')
    
    #how to calculate the covariance that's used for kriging if it's not calculated yet
    cov_choice = config.get('covariance', 'covariance_type')
    
    #check if user specified number of modes for EOF reconstruction if that option chosen
    try:
        ipc = config.getint('covariance', 'pc_number')
    except ValueError:
        ipc = ''
    
    #check type of precessing - daily or pentad
    processing = []
    if config.getboolean('time_resolution', 'daily') == True and config.getboolean('time_resolution', 'pentad') == False:
        print('Daily processing')
        processing = 'daily'
    elif config.getboolean('time_resolution', 'daily') == False and config.getboolean('time_resolution', 'pentad') == True:
        print('Pentad processing')
        processing = 'pentad'
    else:
        raise KeyError('Either no or all time resolutions have been selected in the .ini file')
    
    output_directory = config.get('output', 'output_dir')
    
        
    bnds = [lon_west, lon_east, lat_south, lat_north]
    #extract the latitude and longitude boundaries from user input
    lon_bnds, lat_bnds = (bnds[0], bnds[1]), (bnds[2], bnds[3])
    print(lon_bnds, lat_bnds)
    
    
    #land-water-mask for observations
    water_mask_file = config.getlist('observations', 'land_mask')
    print(water_mask_file)
    #for ellipse Atlatic 0, for ellipse world 1, for ESA world 2
    water_mask_file = water_mask_file[1]
    print(water_mask_file)
    mask_ds, mask_ds_lat, mask_ds_lon = obs_module.landmask(water_mask_file)
    print('----')
    print(mask_ds)
    
    
    year_list = list(range(int(year_start), int(year_stop)+1,1))
    month_list = list(range(1,13,1))
    for i in range(len(year_list)):
        current_year = year_list[i]
        for j in range(len(month_list)):
            current_month = month_list[j]
            
            try:
                ncfile.close()  #make sure dataset is not already open.
            except: 
                pass
            
            ncfilename = str(output_directory) + str(current_year) + str(current_month).zfill(2) + 'pentads_kriged.nc'
            ncfile = nc.Dataset(ncfilename,mode='w',format='NETCDF4_CLASSIC') 
            print(ncfile)
            
            lat_dim = ncfile.createDimension('lat', len(mask_ds_lat))    # latitude axis
            lon_dim = ncfile.createDimension('lon', len(mask_ds_lon))    # longitude axis
            time_dim = ncfile.createDimension('time', None)         # unlimited axis
            
            # Define two variables with the same names as dimensions,
            # a conventional way to define "coordinate variables".
            lat = ncfile.createVariable('lat', np.float32, ('lat',))
            lat.units = 'degrees_north'
            lat.long_name = 'latitude'
            lon = ncfile.createVariable('lon', np.float32, ('lon',))
            lon.units = 'degrees_east'
            lon.long_name = 'longitude'
            time = ncfile.createVariable('time', np.float64, ('time',))
            time.units = 'days since %s-%s-03' % (str(current_year), str(current_month).zfill(2))
            time.long_name = 'time'
            print(time)
            
            
            # Define a 3D variable to hold the data
            ok = ncfile.createVariable('sst_anomaly',np.float64,('time','lat','lon'))
            # note: unlimited dimension is leftmost
            ok.units = 'deg C' # degrees Kelvin
            ok.standard_name = 'SST anomaly'
            # Define a 3D variable to hold the data
            clim = ncfile.createVariable('climatology',np.float64,('time','lat','lon'))
            # note: unlimited dimension is leftmost
            clim.units = 'deg C' # degrees Kelvin
            clim.standard_name = 'SST climatology' # this is a CF standard name
            # Define a 3D variable to hold the data
            dz_ok = ncfile.createVariable('sst_anomaly_uncertainty',np.float64,('time','lat','lon')) # note: unlimited dimension is leftmost
            dz_ok.units = 'deg C' # degrees Kelvin
            dz_ok.standard_name = 'uncertainty' # this is a CF standard name
            
            
            # Write latitudes, longitudes.
            # Note: the ":" is necessary in these "write" statements
            lat[:] = mask_ds_lat #ds.lat.values
            lon[:] = mask_ds_lon #ds.lon.values
            
    
            obs_df = obs_qc_module.main(data_path, qc_path, year=current_year, month=current_month)
            print(obs_df.columns.values)
            
            
            if cov_dir is not None:
                #match covariance to the processed monthly data file
                print(cov_dir)
                covariance = cov_module.read_in_covarance_file(cov_dir, month=current_month)
                #np.fill_diagonal(covariance, [1.01*np.diag(covariance)])
            else:
                #calculate covariance based on ESA CCI anomalies and set up a (land- and ice-) masked ESA CCI dataset
                covariance, ds_masked = cov_module.covariance_main(cci_directory, ds_varname, lon_west, lon_east, lat_south, lat_north, cov_choice, ipc=ipc, time_resolution=False)
            print(covariance)
            """
            #save out the obs dataframe into csv to plot locations of ships?
            unique_days = np.unique(obs_df['dy'])
            print('unique days', unique_days)
            """
            # list of dates for each year 
            #current_month_range = pd.date_range(str(current_month)+'/1/'+str(current_year),str(current_month)+'/31/'+str(current_year), freq='D')
            _,month_range = monthrange(current_year, current_month)
            print(month_range)
        
            #for day in range(1,month_range+1,1): #unique_days)):
            
            # for daily processing
                #day_df = obs_df[obs_df['dy'] == day]
                #timestep = int(day) - 1
                
            for pentad in range(1,7):
                print(current_year, current_month, pentad) #day)
                
                #get a middle day of each pentad to extract climatology (for anomalies)
                if pentad == 1:
                    day = 3
                    day_df = obs_df[(day - 2 <= obs_df['dy']) & (obs_df['dy'] < day+3)]
                elif pentad == 2:
                    day = 8
                    day_df = obs_df[(day - 2 <= obs_df['dy']) & (obs_df['dy'] < day+3)]
                elif pentad == 3:
                    day = 13
                    day_df = obs_df[(day - 2 <= obs_df['dy']) & (obs_df['dy'] < day+3)]
                elif pentad == 4:
                    day = 18
                    day_df = obs_df[(day - 2 <= obs_df['dy']) & (obs_df['dy'] < day+3)]
                elif pentad == 5:
                    day = 23
                    day_df = obs_df[(day - 2 <= obs_df['dy']) & (obs_df['dy'] < day+3)]
                elif pentad == 6:
                    day = 28
                    day_df = obs_df[day - 2 <= obs_df['dy']]
                    
                #for pentad processing
                
                timestep = int(pentad) - 1
                print('---------')
                print(day_df['dy'])
                print('----------')
                print('timestep', timestep)
                
                

                current_date = datetime(current_year,current_month,day)
                print(current_date)
                
                #to match a 365 ESA climatology file
                #in order to calculate SST anomalies for the obs
                if not isleap(current_year):
                    print('year is not leap')
                    DOY = int(current_date.strftime('%j'))
                elif isleap(current_year):
                    print('year is leap')
                    DOY = int(current_date.strftime('%j'))
                    if DOY < 29:
                        DOY = DOY
                    elif DOY >= 29:
                        DOY = DOY - 1
                print('DOY', DOY)
                
                esa_climatology = obs_module.read_climatology(cci_climatology,DOY)
                cropped_clim = esa_climatology.sel(lat=slice(lat_south,lat_north), lon=slice(lon_west,lon_east))
                cropped_clim = cropped_clim.variables['analysed_sst'].values.squeeze()
                cropped_clim = cropped_clim - 273.15
                #climatology comes at a very fine resolution
                #ouput product comes at 1 degree, needs coarsening
                
                
                
                #add climatology value and calculate the SST anomaly
                day_df = obs_module.extract_clim_anom(esa_climatology, day_df)
                #calculate flattened idx based on the ESA landmask file
                #which is compatible with the ESA-derived covariance
                cond_df, obs_flat_idx = obs_module.add_esa_watermask_at_obs_locations(lon_bnds, lat_bnds, day_df, mask_ds, mask_ds_lat, mask_ds_lon)
                mask_ds, mask_ds_lat, mask_ds_lon = obs_module.landmask(water_mask_file)
                
                print(cond_df)
                print(cond_df.columns.values)
                
                #quick temperature check
                print(cond_df['sst'])
                print(cond_df['climatology_sst'])
                print(cond_df['sst_anomaly'])
                
                
                """
                plotting_df = cond_df[['lon', 'lat', 'sst', 'climatology_sst', 'sst_anomaly']]
                lons = plotting_df['lon']
                lats = plotting_df['lat']
                ssts = plotting_df['sst']
                clims = plotting_df['climatology_sst']
                anoms = plotting_df['sst_anomaly']
                
                skwargs = {'s': 5, 'c': 'red'}
                fig = plt.figure(figsize=(10, 5))
                ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
                cp.projected_scatter(fig, ax, lons, lats, skwargs=skwargs, title='ICOADS locations - '+str(pentad)+' pentad')
                plt.show()
                
                skwargs = {'s': 5, 'c': ssts, 'cmap': plt.cm.get_cmap('coolwarm'), 'clim': (-10, 14)}
                ckwargs = {'label': 'SST [deg C]'}
                fig = plt.figure(figsize=(10, 5))
                ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
                cp.projected_scatter(fig, ax, lons, lats, add_colorbar=True, skwargs=skwargs, ckwargs=ckwargs, title="ICOADS measured SST - "+str(pentad)+ ' pentad', land_col='darkolivegreen')
                plt.show()
                
                skwargs = {'s': 5, 'c': clims, 'cmap': plt.cm.get_cmap('coolwarm'), 'clim': (-10, 14)}
                ckwargs = {'label': 'SST [deg C]'}
                fig = plt.figure(figsize=(10, 5))
                ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
                cp.projected_scatter(fig, ax, lons, lats, add_colorbar=True, skwargs=skwargs, ckwargs=ckwargs, title="ESA CCI climatology - "+str(pentad)+' pentad', land_col='darkolivegreen')
                plt.show()
                
                skwargs = {'s': 5, 'c': anoms, 'cmap': plt.cm.get_cmap('coolwarm'), 'clim': (-10, 14)}
                ckwargs = {'label': 'SST [deg C]'}
                fig = plt.figure(figsize=(10, 5))
                ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
                cp.projected_scatter(fig, ax, lons, lats, add_colorbar=True, skwargs=skwargs, ckwargs=ckwargs, title="SST anomalies", land_col='darkolivegreen')
                plt.show()
                """
                
                
                day_flat_idx = cond_df['flattened_idx'][:]
                

                
                obs_covariance, W = obs_module.measurement_covariance(cond_df, day_flat_idx, sig_ms=1.27, sig_mb=0.23, sig_bs=1.47, sig_bb=0.38)
                #W = obs_module.counts_for_esa(day_flat_idx)
                print(obs_covariance)
                print(W)
                
                #krige obs onto gridded field
                obs_sk_2d, dz_sk_2d, obs_ok_2d, dz_ok_2d = krig_module.kriging_main(covariance, mask_ds, cond_df, day_flat_idx, obs_covariance, W)
                #obs_sk_2d, dz_sk_2d, obs_ok_2d, dz_ok_2d = krig_module.kriging_main(covariance, ds_masked, day_df, day_flat_idx,  W)
                print('Kriging done, saving output')
                
                # Write the data.  
                #This writes each time slice to the netCDF instead of the whole 3D netCDF variable all at once.
                ok[timestep,:,:] = obs_ok_2d #ordinary_kriging
                dz_ok[timestep,:,:] = dz_ok_2d #ordinary_kriging
                #clim[timestep,:,:] = cropped_clim
                print("-- Wrote data")
                print(pentad, day)

                
            
                
                
                
                
            # Write time
            #pd.date_range takes month/day/year as input dates
            #for daily processing
            #dates_ = pd.date_range(str(current_month)+'/1/'+str(current_year), str(current_month)+'/'+str(month_range)+'/'+str(current_year), freq='D')
            #for pentad processing
            dates_ = pd.date_range(str(current_month)+'/3/'+str(current_year), str(current_month)+'/28/'+str(current_year), freq='5D')
            
            dates_ = pd.Series([datetime.combine(i, datetime.min.time()) for i in dates_])
            print('dates', dates_)
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
            STOP
    STOP    






if __name__ == '__main__':
    main(sys.argv[1:])

