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

#math tools 
import numpy as np
import math
from scipy.linalg import block_diag

#plotting tools
import matplotlib.pyplot as plt

#timing tools
import timeit

#data handling tools
import pandas as pd
import xarray as xr
import netCDF4 as nc
from functools import partial

#self-written modules (from the same directory)
import covariance_calculation as cov_cal
import covariance as cov_module
import observations as obs_module
import kriging as krig_module







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
    config = ConfigParser()
    #parce existing config file
    config.read(config_file) #('config.ini' or 'three_step_kriging.ini')
    
    print(config)
    
        
    #read values from auxiliary_files section
    #for string use config.get
    #for boolean use config.getboolean
    #for int use config.getint
    #for float use config.getfloat
    
    #location of ESA CCI files for the covariance creation and output grid
    cci_directory = config.get('auxiliary_files', 'esa_cci') #'/noc/mpoc/surface_data/ESA_CCI1deg_pent/ANOMALY/*.nc
    cci_climatology = config.get('auxiliary_files', 'esa_climatology')

    #what are we processing - variable in the files from cci_directory
    ds_varname = config.get('variable_name', 'variable')
    
    #set boundaries for the domain
    lon_west  = config.getfloat('domain', 'lon_west') #-180. 
    lon_east  = config.getfloat('domain', 'lon_east') #180. 
    lat_south = config.getfloat('domain', 'lat_south') #-90.
    lat_north = config.getfloat('domain', 'lat_north') #90. 
    
    #set step size (grid size)
    dlon = config.getfloat('grid', 'dlon') #0.25
    dlat = config.getfloat('grid', 'dlat') #0.25
    
    #location of the ICOADS observation files
    obs_directory = config.get('auxiliary_files', 'observations')
    
    #time period for processing
    if not args.year_start and args.year_stop:
        #start_date
        year_start = config.get('time_period', 'startdate')
        #end_date
        year_stop = config.get('time_period', 'enddate')
    else:
        year_start = args.year_start
        year_stop = args.year_stop
        
    #how we want to calculate the covariance that's used for kriging
    cov_choice = config.get('covariance', 'covariance_type')
    
    #check if user specified number of modes for EOF reconstruction if that option chosen
    try:
        ipc = config.getint('covariance', 'pc_number')
    except ValueError:
        ipc = ''
    
    
        
    bnds = [lon_west, lon_east, lat_south, lat_north]
    #extract the latitude and longitude boundaries from user input
    lon_bnds, lat_bnds = (bnds[0], bnds[1]), (bnds[2], bnds[3])
    print(lon_bnds, lat_bnds)
    
    
    #calculate covariance based on ESA CCI anomalies and set up a (land- and ice-) masked ESA CCI dataset
    covariance, ds_masked = cov_module.covariance_main(cci_directory, ds_varname, lon_west, lon_east, lat_south, lat_north, cov_choice, ipc=ipc, time_resolution=False)
    
    #extra bit
    water_mask = np.copy(ds_masked.values[:,:,0])
    water_mask[~np.isnan(water_mask)] = 1
    water_mask[np.isnan(water_mask)] = 0
    #end of extra bit
    
    #extract obs and create obs covariance
    #obs_filename = '/noc/mpoc/surface_data/ICOADS_NETCDF_R3_ALL/ICOADS_R3.0.0_1950-02.nc'
    #timestep=100
    
    #obs_covariance, W, cond_df, flattened_idx = obs_module.obs_main(obs_filename, lon_bnds, lat_bnds, ds_masked, timestep)
    year_list = list(range(int(year_start), int(year_stop)+1,1))
    for i in range(len(year_list)):
        current_year = year_list[i]
        cond_df, flattened_idx = obs_module.obs_main(obs_directory, lon_bnds, lat_bnds, ds_masked, str(cci_climatology), year=current_year)
    
    
        #save out the obs dataframe into csv to plot locations of ships?
        unique_days = np.unique(cond_df['day'])
        print('unique days', unique_days)
        # list of dates for each year 
        current_year_range = pd.date_range('1/1/'+str(current_year),'12/31/'+str(current_year), freq='D')
        print(current_year_range)
        
        for i in current_year_range: #unique_days)):
            #print(day+1)
            #
            print(i.day, i.month, i.year)
            break
            day_df = cond_df[cond_df['day'] == i.day]
            day_flat_idx = day_df['flattened_idx'][:]
            
            #extra
            #counts = []
            #idx_list = flattened_idx.tolist()
            #print(idx_list)
            #for index in flattened_idx:
            #    c = idx_list.count(index)
            #    counts.append(c)
            #print(idx_list)
            #print(counts)
            #

            print(day_df)
            print(day_df['date'].values[0])
            print(day_flat_idx, len(day_flat_idx))
            #obs_covariance, W = obs_module.measurement_covariance(day_df, day_flat_idx, sig_ms=1.27, sig_mb=0.23, sig_bs=1.47, sig_bb=0.38)
            W = obs_module.counts_for_esa(day_flat_idx)
            #print(obs_covariance)
            #print(W)
            
            #krige obs onto gridded field
            #obs_sk_2d, dz_sk_2d, obs_ok_2d, dz_ok_2d = krig_module.kriging_main(covariance, ds_masked, day_df, day_flat_idx, obs_covariance, W)
            obs_sk_2d, dz_sk_2d, obs_ok_2d, dz_ok_2d = krig_module.kriging_main(covariance, ds_masked, day_df, day_flat_idx,  W)
            print('Kriging done, saving output')
            """
            #save output
            latitude_grid = ds_masked.lat.values
            longitude_grid = ds_masked.lon.values
            var1 = xr.DataArray(obs_sk_2d,dims=("lat", "lon", "time"),coords={"lat": latitude_grid, "lon": longitude_grid},name="simple kriging",)
            var2 = xr.DataArray(obs_ok_2d,dims=("lat", "lon", "time"),coords={"lat": latitude_grid, "lon": longitude_grid},name="ordinary kriging",)
            merged = xr.merge([var1, var2]) #can add var0 with number of obs
            #new_filename = '/noc/users/agfaul/code/100_3eof/pentad_%s.nc' % (cond_df['year'].values[0])
            new_filename = '/noc/users/agfaul/code/jasmin_try/%s.nc' % (day_df['date'].values[0])
            """
        """
        #plt.imshow(np.flipud(var0))
        #plt.title('Original ESA CCI SST anomalies for a chosen timestep (100 timestep)')
        #plt.colorbar()
        #plt.show()
        plt.imshow(np.flipud(var1))
        plt.title('Reconstructed using observation points from %s first pentad and simple kriging' % str(cond_df['year'].values[0]))
        plt.colorbar()
        plt.show()
        plt.imshow(np.flipud(var2))
        plt.title('Reconstructed using observation points from %s first pentad and ordinary kriging' % str(cond_df['year'].values[0]))
        plt.colorbar()
        plt.show()
        """
        #merged.to_netcdf(path=new_filename)







if __name__ == '__main__':
    main(sys.argv[1:])

