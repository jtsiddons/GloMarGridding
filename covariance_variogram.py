import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from math import *
import pandas as pd
import sys, os, re

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.spatial import distance_matrix
from scipy import optimize

from scipy.stats import binned_statistic


from scipy.linalg import solve

import cartopy.crs as ccrs

from tqdm import tqdm

from math import radians, cos, sin, asin, sqrt

#from a different folder:
#import models




#######################
# define any functions here
#######################
def linear_variogram_model(m, d):
    """
    Linear model
    m is a list containing [slope, nugget]
    d is an array of the distance values at which to calculate the variogram model
    from: Code by Benjamin S. Murphy and the PyKrige Developers
bscott.murphy@gmail.com
    """
    slope = float(m[0])
    nugget = float(m[1])
    return slope * d + nugget


def power_variogram_model(m, d):
    """
    Power model
    m is a list containing [scale, exponent, nugget]
    d is an array of the distance values at which to calculate the variogram model
    from: Code by Benjamin S. Murphy and the PyKrige Developers
bscott.murphy@gmail.com
    """
    scale = float(m[0])
    exponent = float(m[1])
    nugget = float(m[2])
    return scale * d**exponent + nugget


def gaussian_variogram_model(m, d):
    """
    Gaussian model
    m is a list containing [psill, range, nugget]
    d is an array of the distance values at which to calculate the variogram model
    from: Code by Benjamin S. Murphy and the PyKrige Developers
bscott.murphy@gmail.com
    """
    psill = float(m[0])
    range_ = float(m[1])
    nugget = float(m[2])
    return psill * (1.0 - np.exp(-(d**2.0) / (range_ * 4.0 / 7.0) ** 2.0)) + nugget


def exponential_variogram_model(m, d):
    """
    Exponential model
    m is a list containing [psill, range, nugget]
    d is an array of the distance values at which to calculate the variogram model
    from: Code by Benjamin S. Murphy and the PyKrige Developers
bscott.murphy@gmail.com
    """
    psill = float(m[0])
    range_ = float(m[1])
    nugget = float(m[2])
    #C(h) = (sill - range) exp (-3 |h|/r), if |h| > 0 #nugget missing here?
    return psill * (1.0 - np.exp(-d / (range_ / 3.0))) + nugget #this should be the correct version



def watermask(ds_masked_xr):
    water_mask = np.copy(ds_masked_xr.values[:,:])
    water_mask[~np.isnan(water_mask)] = 1
    water_mask[np.isnan(water_mask)] = 0
    print(np.shape(water_mask))
    water_idx = np.asarray(np.where(water_mask.flatten() == 1)) #this idx is returned as a row-major        
    print(water_idx)
    return water_idx



def find_values(ds_masked_xr, lat, lon, timestep):
    '''                                                                                                           
    Parameters:                                                                                                   
    cci (array) - array of cci  vaules for each point in the whole domain                                         
    lat (array) - array of latitudes of the observations                                                          
    lon (array) - array of longitudes of the observations                                                         
    df (dataframe) - dataframe containing all information for a given day (such as location, measurement values)  
                                                                                                                  
    Returns:                                                                                                      
    Dataframe with added anomalies for each observation point                                                     
    '''
    cci_lat_idx = find_nearest(ds_masked_xr.lat, lat)
    print(ds_masked_xr.lat, cci_lat_idx)
     #### problem with this right here (see email to Liz and Richard) - for both lat and lon                
    ##### see: https://stackoverflow.com/questions/40592630/get-coordinates-of-non-nan-values-of-xarray-dataset   
    cci_lon_idx = find_nearest(ds_masked_xr.lon, lon)
    print(ds_masked_xr.lon, cci_lon_idx)
    cci_vals = [] #fake ship obs                                                                            
    for i in range(len(cci_lat_idx)):
        #cci_ = cci[str(ds_var)].values[timestep,cci_lat_idx[i],cci_lon_idx[i]] #was cci.sst_anomaly        
        cci_ = ds_masked_xr.values[cci_lat_idx[i],cci_lon_idx[i],timestep] #was cci.sst_anomaly             
        cci_vals.append(cci_)
    cci_vals = np.hstack(cci_vals)
    to_remove = np.argwhere(np.isnan(cci_vals))
    #cci_vals = np.delete(cci_vals, to_remove)                                                              
    cci_lat_idx = np.delete(cci_lat_idx, to_remove)
    cci_lon_idx = np.delete(cci_lon_idx, to_remove)
    return cci_vals, cci_lat_idx, cci_lon_idx




def find_nearest(array, values):
    array = np.asarray(array)
    idx_list = [(np.abs(array - value)).argmin()for value in values]
    return idx_list




def getDistanceByHaversine(loc1, loc2, to_radians=True, earth_radius=6371):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    
    Input: latitude and longitude columns from Pandas dataframe (arrays of values)
    Output: a 2D numpy array of (lat_decimal,lon_decimal) pairs
    """
    # "unpack" our numpy array, this extracts column wise arrays
    lat1 = loc1[0]
    lon1 = loc1[1]
    lat2 = loc2[0]
    lon2 = loc2[1]
    if to_radians:
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula                                                                                   
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    km = earth_radius * 2 * np.arcsin(np.sqrt(a))
    return km



def calculate_distance_matrix(df):
    distance_matrix = pd.DataFrame(squareform(pdist(df, lambda u, v: getDistanceByHaversine(u,v))), index=df.index, columns=df.index)
    return distance_matrix



def variogram(distance_matrix, variance):
    #range from Dave's presentation on space length scales (in km)
    range_ = 350 
    #from researchgate - Sill of the semivariogram is equal to the variance of the random variable, at large distances, when variables become uncorrelated
    sill_ = variance 
    #nugget for now can be set to zero, it will change once we quantify the obs uncertainty better
    nugget_ = 0 
    
    #create m - a list containing [psill, range, nugget]
    m = [sill_, range_, nugget_]
    
    #create d - an array of the distance values at which to calculate the variogram model
    d = distance_matrix
    
    #call variogram function
    #this calculates the covariance between the observations only (equivalemnt in Simon's code is "S" martix
    obs_covariance = exponential_variogram_model(m, d)
    return obs_covariance

"""
######################
# read in the observational data
######################
filename = pd.read_csv(sys.argv[1]) # netcdf file #/noc/users/agfaul/data_nc/MAT_1820_7.nc
lats = filename['lat'].values #np.asarray(nc.Dataset(filename).variables['lat']) #it's a 1-D array based on the psv file
lons = filename['lon'].values #np.asarray(nc.Dataset(filename).variables['lon']) #it's a 1-D array based on the psv file
#coords = list(zip(list(lats),list(lons)))
vals = filename['anomaly'].values #np.asarray(nc.Dataset(filename).variables['anomaly']) #it's a 1-D array based on the psv file

print('LATS', lats, len(lats))
print('LONS', lons, len(lons))
#print('COORDS', coords)
print('VALS', vals, len(vals))



##################
# compute pairwise distances
##################

# distance matrix using pandas
data = {'lat': lats, 'lon': lons}
df = pd.DataFrame(data, columns=['lat', 'lon'])

distance_matrix = pd.DataFrame(squareform(pdist(df, lambda u, v: getDistanceByHaversine(u,v))), index=df.index, columns=df.index)

print('Distance matrix \n', distance_matrix)


#####################
# compute the raw variogram values
#####################
dissim = np.abs(vals[:, None] - vals[None, :])**2 / 2
dissim_ = squareform(dissim)

variance = np.var(vals)
print('VARIANCE', variance)


#####################
# set nugget, sill and range
#####################

#range from Dave's presentation on space length scales (in km)
range_ = 350 
#from researchgate - Sill of the semivariogram is equal to the variance of the random variable, at large distances, when variables become uncorrelated
sill_ = variance 
#nugget for now can be set to zero, it will change once we quantify the obs uncertainty better
nugget_ = 0 

#create m - a list containing [psill, range, nugget]
m = [sill_, range_, nugget_]

#create d - an array of the distance values at which to calculate the variogram model
d = distance_matrix

#call variogram function
#this calculates the covariance between the observations only (equivalemnt in Simon's code is "S" martix
obs_covariance = exponential_variogram_model(m, d)

print(variogram)
print(variogram.shape)


######################
# set up a global grid for distance matrix computation
#####################

#create global grid for finding prediction points                                                          
global_lats = np.arange(-90., 90.,1.)
global_lons = np.arange(-180., 180.,1.)

#calculate the covariance between the observation points and the prediction points (equivlent in Simon's code is the "Ss" matrix

global_data = {'lat': global_lats, 'lon': global_lons}
global_df = pd.DataFrame(global_data, columns=['lat', 'lon'])
global_distance_matrix = pd.DataFrame(distance_matrix(global_df.values, global_df.values), index=global_df.index, columns=global_df.index)
print('global distance_matrix \n', global_distance_matrix)

d_global = global_distance_matrix
global_covariance = exponential_variogram_model(m, d_global)


timestep = 0
ds_masked_xr = xr.DataArray(ds_masked, coords=cci_pentad_files.coords, dims=cci_pentad_files.dims, attrs=cci_pentad_files.attrs)
#on a global 1-D grid find the index of where the observations are
iid = watermask(ds_masked_xr,timestep)
#extract the CCI SST anomaly values corresponding to the shop lat/lon coordinate points                       
#fake_ship_obs = find_nearest(ds.lat.values, ship_lat)                                                      
fake_ship_obs, lat_idx, lon_idx = find_values(ds_masked_xr, ship_lat, ship_lon, timestep)
cond_df['cci_anomalies'] = fake_ship_obs
#in case some of the values are Nan (because covered by ice)                                                
cond_df = cond_df.dropna()
idx_tuple = np.array([lat_idx, lon_idx])
#output_size = (ds[str(ds_var)].values[timestep,:,:]).shape                                                 
output_size = (ds_masked_xr.values[:,:,timestep]).shape
flattened_idx = np.ravel_multi_index(idx_tuple, output_size, order='C') #row-major; order='F') column-major
uind = np.unique(flattened_idx)

_,ia,_ = intersect_mtlb(iid,uind)
obs_unknown_covariance = global_covariance[ia,:]

#the rest should be the same in kriging
#idea: could try the kriging code with weights provided by the https://gis.stackexchange.com/questions/270274/how-to-calculate-kriging-weights/301362#301362 and see how different they are
"""
