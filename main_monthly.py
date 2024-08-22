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
    parser.add_argument("-method", dest="method", default="simple", required=False, help="Kriging Method - one of \"simple\" or \"ordinary\"")
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
    metoffice_climatology = config.get('observations', 'metoffice_climatology')
    monthly_climatology = config.get('observations', 'monthly_climatology')

    #what are we processing - variable in the files from cci_directory
    ds_varname = config.getlist('variable_name', 'variable')
    #for sst 0, for sst_anomaly 1
    ds_varname = ds_varname[1]
    
    #set boundaries for the domain
    lon_west  = config.getfloat('domain', 'lon_west') #-180. 
    lon_east  = config.getfloat('domain', 'lon_east') #180. 
    lat_south = config.getfloat('domain', 'lat_south') #-90.
    lat_north = config.getfloat('domain', 'lat_north') #90. 
    
    #location of the ICOADS observation files
    data_path = config.get('observations', 'observations')
    #location og QC flags in GROUPS subdirectories
    qc_path = config.get('observations', 'qc_flags_joe')
    qc_path_2 = config.get('observations', 'qc_flags_joe_tracked')
    
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
    
    
    
    output_directory = config.get('output', 'output_dir')
    
        
    bnds = [lon_west, lon_east, lat_south, lat_north]
    #extract the latitude and longitude boundaries from user input
    lon_bnds, lat_bnds = (bnds[0], bnds[1]), (bnds[2], bnds[3])
    print(lon_bnds, lat_bnds)
    
    
    #land-water-mask for observations
    water_mask_dir = config.get('covariance', 'covariance_path')
    mask_ds, mask_ds_lat, mask_ds_lon = cov_module.get_landmask(water_mask_dir, month=1)
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
    
    
    climatology = obs_module.read_climatology(monthly_climatology, lat_south, lat_north, lon_west, lon_east)
    print(climatology)
    clim_times = climatology.time
    print(clim_times)
    
    #climatology2 = np.broadcast_to(mask_ds.landmask.values > 0, climatology.climatology.values.shape)
    
    year_list = list(range(int(year_start), int(year_stop)+1,1))
    month_list = list(range(1,13,1))
    for i in range(len(year_list)):
        current_year = year_list[i]
        
        print(climatology)
        try:
            ncfile.close()  #make sure dataset is not already open.
        except: 
            pass
                    
        ncfilename = str(output_directory) + str(current_year) + '_monthly_kriged.nc'
        ncfile = nc.Dataset(ncfilename,mode='w',format='NETCDF4_CLASSIC') 
        #print(ncfile)
        
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
        time.units = 'days since %s-01-01' % (str(current_year))
        time.long_name = 'time'
        #print(time)
        
        # Define a 3D variable to hold the data
        krig_anom = ncfile.createVariable('sst_anomaly',np.float32,('time','lat','lon'))
        # note: unlimited dimension is leftmost
        krig_anom.units = 'deg C' # degrees Kelvin
        krig_anom.standard_name = 'SST anomaly'
        # Define a 3D variable to hold the data
        krig_uncert = ncfile.createVariable('sst_anomaly_uncertainty',np.float32,('time','lat','lon')) # note: unlimited dimension is leftmost
        krig_uncert.units = 'deg C' # degrees Kelvin
        krig_uncert.standard_name = 'uncertainty' # this is a CF standard name
        
        # Write latitudes, longitudes.
        # Note: the ":" is necessary in these "write" statements
        lat[:] = mask_ds_lat #ds.lat.values
        lon[:] = mask_ds_lon #ds.lon.values
        
        
        for j in range(len(month_list)):
            current_month = month_list[j]
            #print(current_month)
            obs_df = obs_qc_module.main(data_path, qc_path, qc_path_2, year=current_year, month=current_month)
            #print(obs_df.columns.values)
            mon_df = obs_df
            
            #covariance = cov_module.read_in_covarance_file(cov_dir, month=current_month)
            covariance = cov_module.get_covariance(cov_dir, month=current_month)
            mask_ds, mask_ds_lat, mask_ds_lon = cov_module.get_landmask(water_mask_dir, month=current_month)

                    
            timestep = j
            current_date = datetime(current_year,current_month,15)
            print('----------')
            print('timestep', timestep)
            print('----------')
            print(current_date)
            
            esa_climatology = climatology['climatology'] #[timestep]
            print(esa_climatology)
            
            #add climatology value and calculate the SST anomaly
            #mon_df = obs_module.extract_clim_anom(esa_climatology, mon_df)
            mon_df = obs_qc_module.SST_match_climatology_to_obs(esa_climatology, mon_df)
            #calculate flattened idx based on the ESA landmask file
            #which is compatible with the ESA-derived covariance
            #mask_ds, mask_ds_lat, mask_ds_lon = obs_module.landmask(water_mask_file, lat_south,lat_north, lon_west,lon_east)
            cond_df, obs_flat_idx = obs_module.watermask_at_obs_locations(lon_bnds, lat_bnds, mon_df, mask_ds, mask_ds_lat, mask_ds_lon)
            
            #print(cond_df.columns.values)
            #print(cond_df[['lat', 'lon', 'flattened_idx', 'sst', 'climatology_sst', 'sst_anomaly']])
            #quick temperature check
            #print(cond_df['sst'])
            #print(cond_df['climatology_sst'])
            #print(cond_df['sst_anomaly'])
                
            """
            plotting_df = cond_df[['lon', 'lat', 'sst', 'climatology_sst', 'sst_anomaly']]
            lons = plotting_df['lon']
            lats = plotting_df['lat']
            ssts = plotting_df['sst']
            clims = plotting_df['climatology_sst']
            anoms = plotting_df['sst_anomaly']
            
            skwargs = {'s': 2, 'c': 'red'}
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            cp.projected_scatter(fig, ax, lons, lats, skwargs=skwargs, title='ICOADS locations - '+str(pentad_idx)+' pentad '+ str(current_year)+' year')
            #plt.show()
            fig.savefig('/noc/users/agfaul/ellipse_kriging/%s_%spoints.png' % (str(current_year), str(pentad_idx)))
            
            skwargs = {'s': 2, 'c': ssts, 'cmap': plt.cm.get_cmap('coolwarm'), 'clim': (-10, 14)}
            ckwargs = {'label': 'SST [deg C]'}
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            cp.projected_scatter(fig, ax, lons, lats, add_colorbar=True, skwargs=skwargs, ckwargs=ckwargs, title='ICOADS measured SST -' +str(pentad_idx)+ ' pentad ' + str(current_year)+' year', land_col='darkolivegreen')
            #plt.show()
            fig.savefig('/noc/users/agfaul/ellipse_kriging/%s_%ssst.png' % (str(current_year), str(pentad_idx)))
            
            skwargs = {'s': 2, 'c': clims, 'cmap': plt.cm.get_cmap('coolwarm'), 'clim': (-10, 14)}
            ckwargs = {'label': 'SST [deg C]'}
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            cp.projected_scatter(fig, ax, lons, lats, add_colorbar=True, skwargs=skwargs, ckwargs=ckwargs, title='ESA CCI climatology - '+str(pentad_idx)+' pentad '+ str(current_year)+' year', land_col='darkolivegreen')
            #plt.show()
            fig.savefig('/noc/users/agfaul/ellipse_kriging/%s_%sclim.png' % (str(current_year), str(pentad_idx)))
            
            skwargs = {'s': 2, 'c': anoms, 'cmap': plt.cm.get_cmap('coolwarm'), 'clim': (-10, 14)}
            ckwargs = {'label': 'SST [deg C]'}
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            cp.projected_scatter(fig, ax, lons, lats, add_colorbar=True, skwargs=skwargs, ckwargs=ckwargs, title='ICOADS SST anomalies - '+str(pentad_idx)+ ' pentad '+ str(current_year)+' year', land_col='darkolivegreen')
            #plt.show()
            fig.savefig('/noc/users/agfaul/ellipse_kriging/%s_%sanom.png' % (str(current_year), str(pentad_idx)))
            """
                
            mon_flat_idx = cond_df['flattened_idx'][:]
            
            obs_covariance, W = obs_module.measurement_covariance(cond_df, mon_flat_idx, sig_ms=1.27, sig_mb=0.23, sig_bs=1.47, sig_bb=0.38)
            #print(obs_covariance)
            #print(W)
            
            #krige obs onto gridded field
            anom, uncert = krig_module.kriging_main(covariance, mask_ds, cond_df, mon_flat_idx, obs_covariance, W, kriging_method=args.method)
            print('Kriging done, saving output')
            """
            fig = plt.figure(figsize=(10, 5))
            img_extent = (-180., 180., -60., 60.)
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            ax.set_extent([-180., 180., -60., 60.], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, color='darkolivegreen')
            ax.coastlines()
            m = plt.imshow(np.flipud(obs_ok_2d), origin='upper', extent=img_extent, transform=ccrs.PlateCarree(), cmap=plt.cm.get_cmap('coolwarm'))
            fig.colorbar(m)
            plt.clim(-4, 4)
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
            gl.xlabels_top = False
            gl.ylabels_right = False
            ax.set_title('Kriged SST anomalies ' +str(pentad_idx)+' pentad '+str(current_year)+' year')
            #plt.show()
            fig.savefig('/noc/users/agfaul/ellipse_kriging/%s_%skriged.png' % (str(current_year), str(pentad_idx)))
            """
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
        """
        #dates_ = pd.date_range(str(current_month)+'/3/'+str(current_year), str(current_month)+'/28/'+str(current_year), freq='5D')
        #dates_ = pd.Series([datetime.combine(i, datetime.min.time()) for i in dates_])
        print('dates', dates_)
        dates = dates_.dt.to_pydatetime() # Here it becomes date
        print('pydate', dates)
        times = date2num(dates, time.units)
        print(times)
        """
        time[:] = times
        print(time)    
        # first print the Dataset object to see what we've got
        print(ncfile)
        # close the Dataset.
        ncfile.close()
        print('Dataset is closed!')
        STOP




if __name__ == '__main__':
    main(sys.argv[1:])
