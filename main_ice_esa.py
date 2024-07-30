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
    cci_climatology = config.get('SST', 'esa_climatology')
    cci_daily_climatology = config.get('SST', 'esa_daily_climatology')
    metoffice_climatology = config.get('SST', 'metoffice_climatology')

    #what are we processing - variable in the files from cci_directory
    ds_varname = config.getlist('variable_name', 'variable')
    #for sst 0, for sst_anomaly 1
    ds_varname = ds_varname[1]
    
    #set boundaries for the domain
    lon_west  = config.getfloat('SST', 'lon_west') #-180. 
    lon_east  = config.getfloat('SST', 'lon_east') #180. 
    lat_south = config.getfloat('SST', 'lat_south') #-90.
    lat_north = config.getfloat('SST', 'lat_north') #90. 
    
    #read measurement and bias uncertainties from config
    sig_ms = config.getfloat('SST', 'sig_ms')
    sig_mb = config.getfloat('SST', 'sig_mb')
    sig_bs =  config.getfloat('SST', 'sig_bs')
    sig_bb = config.getfloat('SST', 'sig_bb')
    
    #location of the ICOADS observation files
    data_path = config.get('observations', 'observations')
    #location og QC flags in GROUPS subdirectories
    qc_path = config.get('SST', 'qc_flags_joe')
    qc_path_2 = config.get('SST', 'qc_flags_joe_tracked')
    
    if args.year_start and args.year_stop:
        year_start = int(args.year_start)
        year_stop = int(args.year_stop)
    else:
        #start_date
        year_start = config.getint('SST', 'startyear')
        #end_date
        year_stop = config.getint('SST', 'endyear')
    print(year_start, year_stop)
    
    #path where the covariance(s) is/are located
    #if single covariance, then full path
    #if several different covariances, then path to directory
    cov_dir = config.get('SST', 'covariance_path')
    
    output_directory = config.get('SST', 'output_dir')
    
     
    bnds = [lon_west, lon_east, lat_south, lat_north]
    #extract the latitude and longitude boundaries from user input
    lon_bnds, lat_bnds = (bnds[0], bnds[1]), (bnds[2], bnds[3])
    print(lon_bnds, lat_bnds)
    
    output_lat = np.arange(lat_bnds[0]+0.5, lat_bnds[-1]+0.5,1)
    output_lon = np.arange(lon_bnds[0]+0.5, lon_bnds[-1]+0.5,1)
    print(output_lat)
    print(output_lon)
    
    #land-water-mask for observations
    #water_mask_dir = config.get('covariance', 'covariance_path')
    #mask_ds, mask_ds_lat, mask_ds_lon = cov_module.get_landmask(water_mask_dir, month=1)
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
    
    
    climatology = obs_module.read_daily_sst_climatology(cci_daily_climatology, lat_south,lat_north, lon_west,lon_east)
    print(climatology)
    pentad_climatology = obs_module.read_climatology(cci_climatology, lat_south,lat_north, lon_west,lon_east)
    clim_times = pentad_climatology.time
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
                    
        ncfilename = str(output_directory) + str(current_year) + '_pentads_kriged_esa.nc'
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
        time = ncfile.createVariable('time', np.float64, ('time',))
        time.units = 'days since %s-01-01' % (str(current_year))
        time.long_name = 'time'
        #print(time)
        
        # Define a 3D variable to hold the data
        ok = ncfile.createVariable('sst_anomaly',np.float32,('time','lat','lon'))
        # note: unlimited dimension is leftmost
        ok.units = 'deg C' # degrees Kelvin
        ok.standard_name = 'SST anomaly'
        # Define a 3D variable to hold the data
        dz_ok = ncfile.createVariable('sst_anomaly_uncertainty',np.float32,('time','lat','lon')) # note: unlimited dimension is leftmost
        dz_ok.units = 'deg C' # degrees Kelvin
        dz_ok.standard_name = 'uncertainty' # this is a CF standard name
        # Define a 3D variable to hold the data
        grid_obs = ncfile.createVariable('observations_per_gridcell',np.float32,('time','lat','lon'))
        # note: unlimited dimension is leftmost
        grid_obs.units = '' # degrees Kelvin
        grid_obs.standard_name = 'Number of observations within each gridcell'
        
        # Write latitudes, longitudes.
        # Note: the ":" is necessary in these "write" statements
        lat[:] = output_lat #ds.lat.values
        lon[:] = output_lon #ds.lon.values
        
        
        for j in range(len(month_list)):
            current_month = month_list[j]
            #print(current_month)
            obs_df = obs_qc_module.main(data_path, qc_path, qc_path_2, year=current_year, month=current_month)
            #print(obs_df.columns.values)
            
            
            #covariance = cov_module.read_in_covarance_file(cov_dir, month=current_month)
            covariance = cov_module.get_covariance(cov_dir, month=current_month)
            print(covariance)
            diag_ind = np.diag_indices_from(covariance)
            covariance[diag_ind] = covariance[diag_ind]*1.02 + 0.005
            print(covariance)

            mask_ds, mask_ds_lat, mask_ds_lon = cov_module.get_landmask(cov_dir, month=current_month)

            # list of dates for each year 
            _,month_range = monthrange(current_year, current_month)
            #print(month_range)

            for pentad_idx, pentad_day in enumerate(range(3,29,5)):
                print(pentad_idx)
                print(pentad_day)
                
                if pentad_idx < 5:
                    day_df = obs_df[(pentad_day - 2 <= obs_df['dy']) & (obs_df['dy'] < pentad_day+3) & (obs_df['mo'] == current_month)]
                elif pentad_idx == 5:
                    day_df = obs_df[(pentad_day - 2 <= obs_df['dy']) & (obs_df['mo'] == current_month)]
                #print(day_df)
                
                timestep = int(pentad_idx+(current_month-1)*6)
                print('----------')
                print('timestep', timestep)
                print('----------')
                current_date = datetime(current_year,current_month,pentad_day)
                #print(current_date)
                #print(climatology)
                try:
                    esa_climatology = climatology['climatology'] #[timestep]
                except KeyError:
                    esa_climatology = climatology['analysed_sst']
                
                #add climatology value and calculate the SST anomaly
                #day_df = obs_module.extract_clim_anom(esa_climatology, day_df)
                day_df = obs_qc_module.SST_match_climatology_to_obs(esa_climatology, day_df)
                
                #calculate flattened idx based on the ESA landmask file
                #which is compatible with the ESA-derived covariance
                #mask_ds, mask_ds_lat, mask_ds_lon = obs_module.landmask(water_mask_file, lat_south,lat_north, lon_west,lon_east)
                cond_df, obs_flat_idx = obs_module.watermask_at_obs_locations(lon_bnds, lat_bnds, day_df, mask_ds, mask_ds_lat, mask_ds_lon)
                
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
                
                day_flat_idx = cond_df['flattened_idx'][:]
                
                cond_df["gridbox"] = day_flat_idx #.values.reshape(-1)
                gridbox_counts = cond_df['gridbox'].value_counts()
                gridbox_count_np = gridbox_counts.to_numpy()
                gridbox_id_np = gridbox_counts.index.to_numpy()
                del gridbox_counts
                water_mask = np.copy(mask_ds.variables['landice_sea_mask'][:,:])
                grid_obs_2d = krig_module.result_reshape_2d(gridbox_count_np, gridbox_id_np, water_mask)
                
                obs_covariance, W = obs_module.measurement_covariance(cond_df, day_flat_idx, sig_ms, sig_mb, sig_bs, sig_bb)
                #print(obs_covariance)
                #print(W)
                
                #krige obs onto gridded field
                obs_sk_2d, dz_sk_2d, obs_ok_2d, dz_ok_2d = krig_module.kriging_main(covariance, cond_df, mask_ds, day_flat_idx, obs_covariance, W)
                #obs_sk_2d, dz_sk_2d, obs_ok_2d, dz_ok_2d = krig_module.kriging_main(covariance, ds_masked, day_df, day_flat_idx,  W)
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
                ok[timestep,:,:] = obs_ok_2d.astype(np.float32) #ordinary_kriging
                dz_ok[timestep,:,:] = dz_ok_2d.astype(np.float32) #ordinary_kriging
                grid_obs[timestep,:,:] = grid_obs_2d.astype(np.float32)
                print("-- Wrote data")
                print(pentad_idx, pentad_day)
        print('Test whether yearly or monthly processing')
        print(ok.shape)
        print(dz_ok.shape)
        
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
        




if __name__ == '__main__':
    main(sys.argv[1:])
