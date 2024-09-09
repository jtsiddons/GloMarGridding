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
from datetime import datetime, timedelta
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
import utils


####
#for plotting
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.cm as cm
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
    cci_climatology = config.get('TAO', 'esa_climatology')
    cci_daily_climatology = config.get('TAO', 'esa_daily_climatology')
    metoffice_climatology = config.get('TAO', 'metoffice_climatology')
    
    #set boundaries for the domain
    lon_west  = config.getfloat('TAO', 'lon_west') 
    lon_east  = config.getfloat('TAO', 'lon_east') 
    lat_south = config.getfloat('TAO', 'lat_south')
    lat_north = config.getfloat('TAO', 'lat_north') 
    
    #read measurement and bias uncertainties from config
    sig_ms = config.getfloat('TAO', 'sig_ms')
    sig_mb = config.getfloat('TAO', 'sig_mb')
    sig_bs =  config.getfloat('TAO', 'sig_bs')
    sig_bb = config.getfloat('TAO', 'sig_bb')
    
    
    #location of the ICOADS observation files
    data_path = config.get('TAO', 'observations')
    
    if args.year_start and args.year_stop:
        year_start = int(args.year_start)
        year_stop = int(args.year_stop)
    else:
        #start_date
        year_start = config.getint('TAO', 'startyear')
        #end_date
        year_stop = config.getint('TAO', 'endyear')
    print(year_start, year_stop)
    
    #path where the covariance(s) is/are located
    #if single covariance, then full path
    #if several different covariances, then path to directory
    cov_dir = config.get('TAO', 'covariance_path')

    landmask_file = config.get('TAO', 'land_mask')
    
    output_directory = config.get('TAO', 'output_dir')
    print(output_directory)
    
    bnds = [lon_west, lon_east, lat_south, lat_north]
    #extract the latitude and longitude boundaries from user input
    lon_bnds, lat_bnds = (bnds[0], bnds[1]), (bnds[2], bnds[3])
    print(lon_bnds, lat_bnds)
    
    output_lat = np.arange(lat_bnds[0]+0.5, lat_bnds[-1]+0.5,1)
    output_lon = np.arange(lon_bnds[0]+0.5, lon_bnds[-1]+0.5,1)
    print(output_lat)
    print(output_lon)
    
    #climatology = obs_module.read_pentad_climatology(cci_climatology, lat_south, lat_north, lon_west,lon_east)
    climatology = obs_module.read_daily_sst_climatology(cci_daily_climatology, lat_south, lat_north, lon_west,lon_east)
    print(climatology)
    pentad_climatology = obs_module.read_climatology(metoffice_climatology, lat_south,lat_north, lon_west,lon_east)
    clim_times = pentad_climatology.time
    print(clim_times)

    mask_ds, mask_ds_lat, mask_ds_lon = cov_module.get_singlefile_landmask(landmask_file)
    
    year_list = list(range(int(year_start), int(year_stop)+1,1))
    month_list = list(range(1,13,1))
    for current_year in year_list:
        
        try:
            ncfile.close()  #make sure dataset is not already open.
        except: 
            pass
                    
        ncfilename = str(output_directory) + str(current_year) + '_pentads_kriged_metoffice.nc'
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
        krig_anom = ncfile.createVariable('sst_anomaly',np.float32,('time','lat','lon'))
        # note: unlimited dimension is leftmost
        krig_anom.units = 'deg C' # degrees Kelvin
        krig_anom.standard_name = 'SST anomaly'
        # Define a 3D variable to hold the data
        krig_uncert = ncfile.createVariable('sst_anomaly_uncertainty',np.float32,('time','lat','lon')) # note: unlimited dimension is leftmost
        krig_uncert.units = 'deg C' # degrees Kelvin
        krig_uncert.standard_name = 'uncertainty' # this is a CF standard name
        # Define a 3D variable to hold the data
        grid_obs = ncfile.createVariable('observations_per_gridcell',np.float32,('time','lat','lon'))
        # note: unlimited dimension is leftmost
        grid_obs.units = '' # degrees Kelvin
        grid_obs.standard_name = 'Number of observations within each gridcell'
        
        # Write latitudes, longitudes.
        # Note: the ":" is necessary in these "write" statements
        lat[:] = output_lat
        lon[:] = output_lon
        
                
        #add MetOffice pentads here
        yr_rng = pd.date_range('1970/01/03','1970/12/31',freq='5D')
        times2 = [j.replace(year=current_year) for j in yr_rng]
        print(times2)
        times_series = pd.Series(times2)
        by_month = list(times_series.groupby(times_series.map(lambda x: x.month)))
        print(by_month)
        
        for current_month in month_list:
            #print(month_list)
            idx, monthly = by_month[current_month-1]
            print(monthly)
            print(idx)
            print(monthly.index)
            #print(current_month)
            pentad_info_df = pd.DataFrame({'pentad_index': monthly.index, 'date': monthly})
            
            obs_df = obs_qc_module.TAO_obs_main(data_path, year=current_year, month=current_month)
            print(obs_df)
            if obs_df is None or obs_df.empty:
                utils.add_empty_layers([krig_anom, krig_uncert, grid_obs],monthly.index,mask_ds.landmask.shape,)
                continue

            #merge index of the pentad processed as information in a new column into the dataframe
            obs_df = obs_df.merge(pentad_info_df, how='inner', on='date')
            print(obs_df)
            
            covariance = cov_module.get_covariance(cov_dir, month=current_month)
            print(covariance)
            diag_ind = np.diag_indices_from(covariance)
            covariance[diag_ind] = covariance[diag_ind]*1.02 + 0.005
            print(covariance)
            
            """
            # NOT USED FOR TAO PROCESSING,
            #read in ellipse parameters file corresponding to the processed file
            month_ellipse_param = obs_qc_module.ellipse_param(ellipse_param_path, month=current_month, var='SST')
            """
            # list of dates for each year 
            _,month_range = monthrange(current_year, current_month)
            #print(month_range)
            
            #if we do MetOffice processing:
            for timestep in monthly.index:
                print('i', timestep, 'monthly.index', monthly.index)
                pentad_date = monthly[timestep]
                
                day_df = obs_df.loc[(obs_df['date'] == str(pentad_date))]
                print(day_df)

                # ADD CHECK IF DF EMPTY HERE
                if day_df.empty:
                    utils.add_empty_layers([krig_anom, krig_uncert, grid_obs],timestep,mask_ds.landmask.shape,)
                    continue
                
                try:
                    #this needs to be changed to pentad, not daily
                    metoffice_climatology = pentad_climatology['climatology'] #[timestep]
                except KeyError:
                    metoffice_climatology = pentad_climatology['analysed_sst']
                    print(metoffice_climatology)
                    
                #add climatology value and calculate the SST anomaly
                #day_df = obs_module.extract_clim_anom(metoffice_climatology, day_df)
                day_df = obs_qc_module.TAO_match_climatology_to_obs(metoffice_climatology, day_df)
                
                print(day_df)
                #calculate flattened idx based on the ESA landmask file
                #which is compatible with the ESA-derived covariance
                #mask_ds, mask_ds_lat, mask_ds_lon = obs_module.landmask(water_mask_file, lat_south,lat_north, lon_west,lon_east)
                cond_df, obs_flat_idx = obs_module.watermask_at_obs_locations(lon_bnds, lat_bnds, day_df, mask_ds, mask_ds_lat, mask_ds_lon)
                #reset row index in the dataframe after dropping NaNs
                cond_df.reset_index(drop=True, inplace=True)
                print(cond_df)
                print(f'{mask_ds =}')
                
                #print(cond_df.columns.values)
                #print(cond_df[['lat', 'lon', 'flattened_idx', 'sst', 'climatology_sst', 'sst_anomaly']])
                #quick temperature check
                #print(cond_df['sst'])
                #print(cond_df['climatology_sst'])
                #print(cond_df['sst_anomaly'])
                
                """
                #EXTRA BIT FOR PLOTTING INSIDE MAIN CODE
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
                print(day_flat_idx)
                
                """
                # NOT USED FOR TAO
                #match gridded observations to ellipse parameters
                cond_df = obs_module.match_ellipse_parameters_to_gridded_obs(month_ellipse_param, cond_df, mask_ds)
                """
                cond_df["gridbox"] = day_flat_idx #.values.reshape(-1)
                
                gridbox_counts = cond_df['gridbox'].value_counts()
                gridbox_count_np = gridbox_counts.to_numpy()
                gridbox_id_np = gridbox_counts.index.to_numpy()
                del gridbox_counts
                water_mask = np.copy(mask_ds.variables['landmask'][:,:])
                grid_obs_2d = krig_module.result_reshape_2d(gridbox_count_np, gridbox_id_np, water_mask)
                """
                #NOT USED FOR TAO
                #obs_module.match_hadsst_bias_to_gridded_obs(hadsst_bias_month, day_flat_idx, mask_ds)
                """
                
                obs_covariance, W = obs_module.TAO_measurement_covariance(cond_df, day_flat_idx, sig_ms, sig_mb, sig_bs, sig_bb)
                #print(obs_covariance)
                #print(W)
                
                #krige obs onto gridded field
                anom, uncert = krig_module.kriging_main(covariance, cond_df, mask_ds, day_flat_idx, obs_covariance, W, bias=False, kriging_method=args.method)
                print('Kriging done, saving output')
                
                """
                fig = plt.figure(figsize=(10, 5))
                img_extent = (-180., 180., -90., 90.)
                ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
                ax.set_extent([-180., 180., -90., 90.], crs=ccrs.PlateCarree())
                ax.add_feature(cfeature.LAND, color='darkolivegreen')
                ax.coastlines()
                m = plt.imshow(np.flipud(obs_ok_2d), origin='upper', extent=img_extent, transform=ccrs.PlateCarree()) #, cmap=cm.get_cmap('coolwarm'))
                fig.colorbar(m)
                plt.clim(-4, 4)
                gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
                gl.xlabels_top = False
                gl.ylabels_right = False
                ax.set_title('Kriged SST anomalies ' +str(pentad_idx)+' pentad '+str(current_year)+' year')
                plt.show()
                #fig.savefig('/noc/users/agfaul/ellipse_kriging/%s_%skriged.png' % (str(current_year), str(pentad_idx)))
                """
                
                # Write the data.  
                #This writes each time slice to the netCDF
                krig_anom[timestep,:,:] = anom.astype(np.float32) #ordinary_kriging
                krig_uncert[timestep,:,:] = uncert.astype(np.float32) #ordinary_kriging
                grid_obs[timestep,:,:] = grid_obs_2d.astype(np.float32)
                print("-- Wrote data")
                print(timestep, pentad_date)
                
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
        dates_ = clim_times.to_series()
        dates_ = [j.replace(year=current_year) for j in dates_]
        dates = dates_.dt.to_pydatetime() # Here it becomes date
        print('pydate', dates)
        times = date2num(dates, time.units)
        print(times)
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
