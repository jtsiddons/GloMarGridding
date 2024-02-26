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








def _preprocess(x, lon_bnds, lat_bnds):
    return x.sel(lon=slice(*lon_bnds), lat=slice(*lat_bnds))



def time_average(ds, avg_type):
    """                                                                                                  
    #Note, that calling ds.groupby('time.year').mean('time') will be incorrect if you are working with monthly and not daily data. 
    Taking the mean will place equal weight on months of different length, e.g., Feb and July, which is wrong.                                                                            
    #https://stackoverflow.com/questions/39985059/compute-annual-mean-using-x-arrays                     
                                                                                                         
    month_length = ds.time.dt.days_in_month                                                              
    print('month lengths', month_length)                                                                 
                                                                                                         
    pentads_in_month = [int(i/5) for i in month_length]                                                  
    print('number of pentads in each month', pentads_in_month)                                           
                                                                                                         
    # Calculate the weights by grouping by 'time.season'.                                                
    weights = (month_length.groupby("time.season") / month_length.groupby("time.season").sum())          
                                                                                                         
    # Test that the sum of the weights for each season is 1.0                                            
    np.testing.assert_allclose(weights.groupby("time.season").sum().values, np.ones(4))                  
                                                                                                         
    # Calculate the weighted average                                                                     
    ds_weighted = (ds * weights).groupby("time.season").sum(dim="time")                                  
    """

    """                                                                                                  
    Grouby-Related: Resample, Rolling, Coarsen, Resample                                                 
    Resample in xarray is nearly identical to Pandas. It can be applied only to time-index dimensions. Here we compute the five-year mean.                                                                       
    It is effectively a group-by operation, and uses the same basic syntax. Note that resampling changes the length of the the output arrays.                                                                    
    e.g. ds_anom_resample = ds_anom.resample(time='5Y').mean(dim='time')                                 
    """
    if avg_type == 'seasonal':
        ds_avg = ds.groupby("time.season").mean(dim="time")
    elif avg_type == 'monthly':
        ds_avg = ds.groupby("time.month").mean(dim="time")
    elif avg_type == 'monthly_for_each_year':
        ds_avg = ds.resample(time='MS').mean(dim="time")
    elif avg_type == 'yearly':
        ds_avg = ds.groupby("time.year").mean(dim="time")
    elif avg_type == 'seasonal_for_each_year':
        ds_avg = ds.resample(time='QS-DEC').mean(dim="time")
    return ds_avg



def read_in_dataset(Wlon, Elon, Slat, Nlat, path=None):
    bnds = [Wlon, Elon, Slat, Nlat]
    print(bnds)

    #extract the latitude and longitude boundaries from user input                                       
    lon_bnds, lat_bnds = (bnds[0], bnds[1]), (bnds[2], bnds[3])
    print(lon_bnds, lat_bnds)

    #adding a pre-processing function to subselect only that region when extracting the data from the path                                                                                                       
    partial_func = partial(_preprocess, lon_bnds=lon_bnds, lat_bnds=lat_bnds)

    if path is not None:
        print('The path to files for creating the covariance has been specified by the user')
        print('Creating covariance from the files in:' , str(path))
    else:
        print('The path has not been specified by the user, using ESA CCI 1 degree pentad anomalies dataset')
        path = '/noc/mpoc/surface_data/ESA_CCI1deg_pent/ANOMALY/*.nc'
    
    #open all .nc files in the directory and concatenate against time and subselect region based on lats and lons provided
    ds = xr.open_mfdataset(str(path), combine='nested', concat_dim='time', preprocess=partial_func, engine="netcdf4")
    print(ds)
    return ds



def time_average_ds(ds, ds_varname, time_resolution=False):
    print('time', ds.time)
    #extract variable of interest - SST anomaliesfrom functools import partial  
    
    cci_pentad_files = ds[str(ds_varname)]
    ds_avg = ds.groupby("time.month") #.mean(dim="time")
    print(ds_avg[1].time.values)
    
    print(cci_pentad_files.values)
    if time_resolution:
        cci_pentad_files = time_average(cci_pentad_files, str(time_resolution))
    print(cci_pentad_files)
    print(cci_pentad_files.values)
    return cci_pentad_files



def latlon_to_1d(ds):
    lat = ds.variables['lat'][:].values
    lon = ds.variables['lon'][:].values
    print(lat)
    print(lon)
    gridlon, gridlat = np.meshgrid(lon, lat)
    print(gridlat)
    print(gridlon)
    lat_1d = gridlat.flatten(order='C') #same order of reshaping for lat,lon and data
    lon_1d = gridlon.flatten(order='C')#same order of reshaping for lat, lon and data
    return lat_1d, lon_1d



def data_to_2d(cci_pentad_files, time_resolution=False):
    #the file dimensions are (time, lat, lon)          
    #flatten the 2D (lat/lon) field into 1D
    # to do it, swap dimensions to have lats and lons first and then time dimension                      
    try:
        cci_pentad_files = cci_pentad_files.transpose("lat", "lon", "time")
    except ValueError:
        if time_resolution == 'seasonal':
           cci_pentad_files = cci_pentad_files.transpose("lat", "lon", "season")
        elif time_resolution == 'monthly':
            cci_pentad_files = cci_pentad_files.transpose("lat", "lon", "month")
        elif time_resolution == 'yearly':
            cci_pentad_files = cci_pentad_files.transpose("lat", "lon", "year")
    cci_pentad_files_3d_np = cci_pentad_files.to_numpy()
    #flatten the 2D lat/lon field into a 1D array while keeping time dimension intact                    
    pentad_1d_time = cci_pentad_files_3d_np.reshape(-1, cci_pentad_files_3d_np.shape[-1], order='C') #order='F for Fortran style, column-major  
    return pentad_1d_time, cci_pentad_files



def mask_ice_land(pentad_1d_time, cci_pentad_files):
    cci_pentad_files_3d_np = cci_pentad_files.to_numpy()
    #if our set-up is space first, time last (i.e. lat, lon, time) then we essentially want to do a row-major reshape so get all lat/lons in the first row and move rows by each time step                       
    #if our set-up is time first, space last (i.e. time, lat,lon) then we essentially want to do a column-major reshape so get all time steps as all rows of first column and then expand for other spatial points                                                                          
    pentad_1d_no_nans, ds_masked = option_ice_nan(pentad_1d_time, cci_pentad_files)
    #timestep = 25                  
    #pentad_1d_no_nans, cci_mask = option_scene_specific(cci_pentad_files_np, timestep)
    #calculate covariance between points in space and time                    
    #covariance = np.cov(pentad_1d_no_nans)                  
    return pentad_1d_no_nans, ds_masked



def option_ice_nan(pentad_1d_time, cci_pentad_files):
    cci_pentad_files_3d_np = cci_pentad_files.to_numpy()
    print(cci_pentad_files_3d_np.shape)
    # 1 OPTION:                
    #remove nans (for each 1D flattened spatial domain)                 
    # Delete all rows with all NaN value                  
    
    booleanIndex = [not np.any(i) for i in np.isnan(pentad_1d_time)]
    pentad_2d_no_nans = pentad_1d_time[booleanIndex]
    #then create a mask for the observation processing with temporary ice masked out                     
    cci_mask = np.copy(pentad_1d_time)
    print(cci_mask.shape)
    for line in cci_mask:
        mask = np.isnan(line).any()
        line[mask] = np.nan
        #line[~mask] = 1.                                                                                
    ds_masked = np.empty(cci_pentad_files.shape)
    #print(ds_masked)
    for timestep in range(len(cci_mask[0,:])):
        ds_masked[:,:,timestep] = np.reshape(cci_mask[:,timestep], cci_pentad_files_3d_np[:,:,timestep].shape, order='C')
    ds_masked_xr = xr.DataArray(ds_masked, coords=cci_pentad_files.coords, dims=cci_pentad_files.dims, attrs=cci_pentad_files.attrs)
    #print(ds_masked_xr)
    return pentad_2d_no_nans, ds_masked_xr



def option_ice_1_8(pentad_1d_time, cci_pentad_files):
    cci_pentad_files_3d_np = cci_pentad_files.to_numpy()
    # 2 OPTION:                                                                                          
    #remove nans that are always nans (land and forever ice)                                             
    # set "temporary" ice to -1.8 and include in the covariance making                                   
    booleanIndex = [not np.all(i) for i in np.isnan(pentad_1d_time)]
    pentad_1d_no_nans_ = pentad_1d_time[booleanIndex]
    pentad_1d_no_nans = np.nan_to_num(pentad_1d_no_nans_, nan=-1.8)
    #then create a mask for the observation processing with temporary ice masked out                     
    cci_mask = np.copy(pentad_1d_time)
    for line in cci_mask:
        mask = np.isnan(line).all()
        line[mask] = -999
        mask2 = np.isnan(line).any()
        line[mask2] = -1.8
    #cci_mask = np.nan_to_num(cci_mask, nan=-1.8)                                                        
    cci_mask[cci_mask == -999 ] = np.nan

    for timestep in range(len(cci_mask[0,:])):
        ds_masked[:,:,timestep] = np.reshape(cci_mask[:,timestep], cci_pentad_files_np[:,:,timestep].shape, order='C')
    ds_masked_xr = xr.DataArray(ds_masked, coords=cci_pentad_files.coords, dims=cci_pentad_files.dims, attrs=cci_pentad_files.attrs)
    print(ds_masked_xr)
    return pentad_1d_no_nans, ds_masked_xr



def option_scene_specific(cci_pentad_files, timestep):
    cci_pentad_files_3d_np = cci_pentad_files.to_numpy()
    #3 OPTION                                                                                            
    timestep_mask = np.isnan(np.copy(cci_pentad_files_3d_np[:,:,timestep]))
    
    for i in range(cci_pentad_files_np.shape[2]):
        time_layer = cci_pentad_files_np[:,:,i]
        time_layer[timestep_mask] = np.nan
    pentad_1d_time = cci_pentad_files_np.reshape(-1, cci_pentad_files_3d_np.shape[-1], order='C')
        
    #remove nans that are always nans (land and forever ice)                                             
    # set "temporary" ice to -1.8 and include in the covariance making                                   
    booleanIndex = [not np.all(i) for i in np.isnan(pentad_1d_time)] #np.all changed to np.any           
    pentad_1d_no_nans_ = pentad_1d_time[booleanIndex]
    pentad_1d_no_nans = np.nan_to_num(pentad_1d_no_nans_, nan=-1.8)
    #then create a mask for the observation processing with temporary ice masked out                     
    cci_mask = np.copy(cci_pentad_files_3d_np[:,:,timestep])
    for line in cci_mask:
        mask = np.isnan(line).all()
        line[mask] = -999
        mask2 = np.isnan(line).any()
        line[mask2] = -1.8
    #cci_mask = np.nan_to_num(cci_mask, nan=-1.8)                                                        
    cci_mask[cci_mask == -999 ] = np.nan
    #cci_mask2 = np.reshape(cci_mask[:,0], cci_pentad_files[:,:,0].shape, order='C')                     
    for i in range(100):
        cci_mask2 = np.reshape(cci_mask[:,i], cci_pentad_files_3d_np[:,:,i].shape, order='C')
    return pentad_1d_no_nans, cci_mask



def calculate_covariance(pentad_1d_time, lat_1d, lon_1d, cov_choice, ipc):
    if cov_choice == 'empirical':
        covariance = cov_cal.calculate_empirical_covariance(pentad_1d_time, lat_1d, lon_1d)
    elif cov_choice == 'reduced_space':
        if not ipc:
            print('Number of PC has not been provided by the user, using the number suggested by the algorithm to account for 70% of variance')
            covariance = cov_cal.reconstruct_covariance_from_eofs(pentad_1d_time, lat_1d, lon_1d)
        else:
            ipc = int(ipc)
            covariance = cov_cal.reconstruct_covariance_from_eofs(pentad_1d_time, lat_1d, lon_1d, ipc)
    elif cov_choice == 'variogram':
        variance = 4.0 #should be the variance calculated from the observations 
        covariance = cov_cal.covariance_from_variogram(pentad_1d_time, lat_1d, lon_1d, variance)
    return covariance






#######################################################################################
def covariance_main(path, ds_varname, Wlon, Elon, Slat, Nlat, cov_choice, ipc=False, time_resolution=False):
    ds = read_in_dataset(Wlon, Elon, Slat, Nlat, path)
    cci_files = time_average_ds(ds, ds_varname, time_resolution=False)
    lat_1d, lon_1d = latlon_to_1d(ds)
    data_2d, cci_files = data_to_2d(cci_files, time_resolution=False)
    data_1d_no_nans, ds_masked = mask_ice_land(data_2d, cci_files)
    covariance = calculate_covariance(data_2d, lat_1d, lon_1d, cov_choice, ipc)
    return covariance, ds_masked






def read_in_covarance_file(path, month):
    #for a path to a directory with covariances
    if os.path.isdir(path):
        monthDict={1:'january', 2:'february', 3:'march', 4:'april', 5:'may', 6:'june', 7:'july', 8:'august', 9:'september', 10:'october', 11:'november', 12:'december'}
        long_filelist = []
        
        filelist = os.listdir(thedirectory) #os.path.join(thedirectory, thefile)
        print(filelist)
        
        mon_str = monthDict[int(month)]
        print('Matching global covariance for %s' % mon_str)
        r = re.compile('world_' +str(mon_str) + '\w+.nc')
        filtered_list = list(filter(r.match, filelist))
        fullpath_list = [os.path.join(thedirectory,f) for f in filtered_list]
        print(filtered_list)
        print(fullpath_list)
        #long_filelist.extend(fullpath_list)
        #print(long_filelist)
        
        ds = xr.open_dataset(fullpath_list[0], engine="netcdf4")
        print(ds)
    #for a path to a single covariance file
    elif os.path.isfile(path):
        #for a single file covariance
        ds = xr.open_dataset(str(path), engine="netcdf4")
    #print(ds)
    covariance =ds.variables['covariance'].values
    print(covariance)
    return covariance




"""
if __name__ == "__main__":
    covariance_main(path, ds_varname, Wlon, Elon, Slat, Nlat, cov_choice, ipc=False, time_resolution=False)
"""
