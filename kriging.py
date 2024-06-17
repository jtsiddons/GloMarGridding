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








def intersect_mtlb(a, b):
    '''
    Returns data common between two arrays, a and b, in a sorted order and index vectors for a and b arrays
    Reproduces behaviour of Matlab's intersect function

    Parameters:
    a (array) - 1-D array
    b (array) - 1-D array

    Returns:
    1-D array, c, of common values found in two arrays, a and b, sorted in order
    List of indices, where the common values are located, for array a
    List of indices, where the common values are located, for array b
    '''
    a1, ia = np.unique(a, return_index=True)
    b1, ib = np.unique(b, return_index=True)
    aux = np.concatenate((a1, b1))
    aux.sort()
    c = aux[:-1][aux[1:] == aux[:-1]]
    return c, ia[np.isin(a1, c)], ib[np.isin(b1, c)]  



def krige(iid, uind, W, x_obs, cci_covariance, covx, bias=False, clim=False):
    '''
    Returns arrays of krigged observations and anomalies for all grid points in the domain
    
    Parameters:
    iid (list) - ID of all measurement points for the chosen date
    x_obs (list) - all point observations for the chosen date
    clim (list) - climatology values for all observation points
    bias (list) - bias for all observation points
    covx (array) - measurement covariance matrix
    cci_covariance (array) - covariance of all CCI grid points (each point in time and all points against each other)
    df (dataframe) - dataframe containing all information and observations for a chosen date

    Returns:
    Full set of values for the whole domain derived from observation points using Simple Kriging
    Uncertainty associated using Simple Kriging
    Full set of values for the whole domain derived from observation points using Ordinary Kriging
    Uncertainty associated using Ordinary Kriging
    '''
    #So Now we can get to the part where we krige the grid 
    #bias = np.array(bias)
    #bias[np.isnan(bias)] = 0
    if iid.ndim > 1:
        iid = np.squeeze(iid)
    print('iid', iid.shape)
    print('uind', uind.shape)
    _,ia,_ = intersect_mtlb(iid,uind)
    ia = ia.astype(int)
    #print(f'{ia.shape = }') # ia.shape)
    
    #ICOADS obs  
    #print(W)
    #print(x_obs)
    sst_obs = W @ x_obs #- clim[ia] - bias[ia] 
    print('SST OBS', sst_obs.shape)
    
    #R is the covariance due to the measurements i.e. measurement noise, bias noise and sampling noise 
    #takes the ICOADS points covariance and maps to grid point covariance 
    Wtrans = np.transpose(W)
    R  = W @ covx @ Wtrans
    #print('W', W)
    #print('W trans', Wtrans)
    #print('R', R)
    #S is the spatial covariance between all "measured" grid points 
    covar = np.copy(cci_covariance)
    S  = covar[ia[:,None],ia[None,:]]
    print('S', S.shape)
    #Ss is the covariance between to be "predicted" grid points (i.e. all) and "measured" points 
    Ss = covar[ia,:]      
    print('Ss', Ss.shape)
    
    #S+R because the measurements have uncertainties as well as spatial covarince 
    #G is the weight vector for Simple Kriging 
    G = np.transpose(Ss) @ np.linalg.inv(S+R)
    z_obs_sk = G @ sst_obs
    #print('G', G)
    #print('z obs sk', z_obs_sk)
    CG = G @ Ss
    #print('CG', CG)
    diagonal = np.diag(covar-CG)
    #diagonal[abs(diagonal) < 1e-15] = 0.0
    dz_sk = np.sqrt(diagonal)
    dz_sk[np.isnan(dz_sk)] = 0.0
    #print('dz_sk', dz_sk)
    #print('Simple Kriging Done')
    
    #Now we will convert to ordinary kriging 
    S_ = np.concatenate((S+R, np.ones((len(ia),1))), axis=1)
    S= np.concatenate((S_, np.ones((1,len(ia)+1))), axis=0)
    #add a Lagrangian multiplier
    S[-1, -1] = 0

    Ss = np.concatenate((Ss, np.ones((1,len(iid)))), axis = 0)
    
    G = np.transpose(Ss) @ np.linalg.inv(np.matrix(S))
    CG = G @ Ss 
    
    sst_obs0 = np.append(sst_obs, 0)
    z_obs_ok = np.transpose(G @ sst_obs0) 
    
    alpha = G[:,-1]
  
    diagonal = (np.diag(covar-CG)).reshape(-1,1)
    dz_ok = np.sqrt(diagonal-alpha)

    print('Ordinary Kriging Done')
    #get rid of resulting double brackets
    a = np.squeeze(np.asarray(z_obs_sk))
    b = np.squeeze(np.asarray(dz_sk))
    c = np.squeeze(np.asarray(z_obs_ok))
    d = np.squeeze(np.asarray(dz_ok))
    return a, b, c, d




def result_reshape_2d(result_1d, iid, grid_2d):
    '''
    Returns reshaped krigged output array, from 1-D into 2-D reproducing Matlab's functionality (going over all rows first, before moving onto columns)

    Parameters:
    result_1d (array) - krigged output array in 1-D
    iid (array) - mask array indicating locations of water regions in the domain
    grid_2d (array) - 2-D array of the output domain

    Returns:
    Krigged result over water areas on a 2-D domain grid with masked land areas
    '''
    result_1d = result_1d.astype('float')
    grid_2d = grid_2d.astype('float')
    
    if grid_2d.ndim > 1:
        grid_2d_flat = grid_2d.flatten()

    to_modify = grid_2d_flat

    if iid.ndim > 1:
        iid = np.squeeze(iid)
    indexes = iid

    landmask = np.copy(to_modify)
    #print(landmask)
    
    if(~np.isnan(landmask).any()):
        landmask = landmask.astype('float')
        landmask[landmask == 0] = np.nan

    replacements = result_1d
    for (index, replacement) in zip(indexes, replacements):
      to_modify[index] = replacement
    to_modify = to_modify * landmask
    result_2d = np.reshape(to_modify, (grid_2d.shape))
    return(result_2d)



def watermask(ds_masked):
    try:
        water_mask = np.copy(ds_masked.variables['landmask'][:,:])
    except KeyError:
        #water_mask = np.copy(ds_masked.variables['land_sea_mask'][:,:])
        water_mask = np.copy(ds_masked.variables['landice_sea_mask'][:,:])
    """
    water_mask[~np.isnan(water_mask)] = 1
    water_mask[np.isnan(water_mask)] = 0
    """
    #print(np.shape(water_mask))
    water_idx = np.asarray(np.where(water_mask.flatten() == 1)) #this idx is returned as a row-major
    #water_idx = np.asarray(np.where(water_mask.flatten(order='F') == 1)) #this idx is returned as a column-major
    #print(water_idx)
    return water_mask, water_idx



def kriged_output(covariance, cond_df, ds_masked, flattened_idx, obs_cov, W):
    try:
        obs = cond_df['sst_anomaly'].values #cond_df['cci_anomalies'].values
    except KeyError:
        obs = cond_df['obs_anomalies_height'].values
    #print('1 - DONE')
    #water_mask, water_idx = watermask(ds, ds_var, timestep)
    water_mask, water_idx = watermask(ds_masked)
    obs_idx = flattened_idx
    unique_obs_idx = np.unique(obs_idx)
    #print('2 - DONE')
   # _,ia,_ = intersect_mtlb(water_idx,unique_obs_idx)
    #W = np.zeros((int(max(unique_obs_idx.shape)),int(max(obs_idx.shape))))
    #for k in range(max(unique_obs_idx.shape)):
        #q = [i for i, x in enumerate(obs_idx) if x == unique_obs_idx[k]]
        #for i in range(len(q)):
            #qq = q[i]
            #W[k,qq] = np.divide(1, len(q))
    obs_sk, dz_sk, obs_ok, dz_ok = krige(water_idx, unique_obs_idx, W, obs, covariance, obs_cov)
    #print('3 - DONE')
    obs_sk_2d = result_reshape_2d(obs_sk, water_idx, water_mask)
    #print('4 - DONE')
    dz_sk_2d = result_reshape_2d(dz_sk, water_idx, water_mask)
    #print('5 - DONE')
    obs_ok_2d = result_reshape_2d(obs_ok, water_idx, water_mask)
    #print('6 - DONE')
    dz_ok_2d = result_reshape_2d(dz_ok, water_idx, water_mask)
    #print('7 - DONE')
    """
    plt.imshow(obs_sk_2d)
    plt.show()
    plt.imshow(dz_sk_2d)
    plt.show()
    
    plt.imshow(obs_ok_2d)
    plt.show()
    plt.imshow(dz_ok_2d)
    plt.show()
    """
    return obs_sk_2d, dz_sk_2d, obs_ok_2d, dz_ok_2d




def krige_for_esa_values_only(iid, uind, W, x_obs, cci_covariance, bias=False, clim=False):
    #this basicallt removes the R matrix (which is the obs covariance that has noise, uncertainty etc) as we use actuall ESA CCI SST anomaly values that we take as ground truth and therefore we assume they have no uncertainty or bias
    '''
    Returns arrays of krigged observations and anomalies for all grid points in the domain
    
    Parameters:
    iid (list) - ID of all measurement points for the chosen date
    x_obs (list) - all point observations for the chosen date
    clim (list) - climatology values for all observation points
    bias (list) - bias for all observation points
    covx (array) - measurement covariance matrix
    cci_covariance (array) - covariance of all CCI grid points (each point in time and all points against each other)
    df (dataframe) - dataframe containing all information and observations for a chosen date

    Returns:
    Full set of values for the whole domain derived from observation points using Simple Kriging
    Uncertainty associated using Simple Kriging
    Full set of values for the whole domain derived from observation points using Ordinary Kriging
    Uncertainty associated using Ordinary Kriging
    '''
    #So Now we can get to the part where we krige the grid 
    #bias = np.array(bias)
    #bias[np.isnan(bias)] = 0
    if iid.ndim > 1:
        iid = np.squeeze(iid)
    print('iid', iid)
    print('uind', uind)
    _,ia,_ = intersect_mtlb(iid,uind)
    ia = ia.astype(int)
    print('ia', ia)
    
    #ICOADS obs  
    sst_obs = W @ x_obs #- clim[ia] - bias[ia] 
    print('SST OBS', sst_obs)
    
   
    #S is the spatial covariance between all "measured" grid points 
    covar = np.copy(cci_covariance)
    S  = covar[ia[:,None],ia[None,:]]
    print('S', S)
    
    #Ss is the covariance between to be "predicted" grid points (i.e. all) and "measured" points 
    Ss = covar[ia,:]      
    print('Ss', Ss)
    
    #G is the weight vector for Simple Kriging 
    G = np.transpose(Ss) @ np.linalg.inv(S)
    z_obs_sk = G @ sst_obs
    print('G', G)
    print('z obs sk', z_obs_sk)
    CG = G @ Ss
    print('CG', CG)
    diagonal = np.diag(covar-CG)
    #diagonal[abs(diagonal) < 1e-15] = 0.0
    dz_sk = np.sqrt(diagonal)
    dz_sk[np.isnan(dz_sk)] = 0.0
    print('dz_sk', dz_sk)
    print('Simple Kriging Done')
    
    #Now we will convert to ordinary kriging 
    S_ = np.concatenate((S, np.ones((len(ia),1))), axis=1)
    S= np.concatenate((S_, np.ones((1,len(ia)+1))), axis=0)
    #add a Lagrangian multiplier
    S[-1, -1] = 0

    Ss = np.concatenate((Ss, np.ones((1,len(iid)))), axis = 0)
    
    G = np.transpose(Ss) @ np.linalg.inv(np.matrix(S))
    CG = G @ Ss 
    
    sst_obs0 = np.append(sst_obs, 0)
    z_obs_ok = np.transpose(G @ sst_obs0) 
    
    alpha = G[:,-1]
  
    diagonal = (np.diag(covar-CG)).reshape(-1,1)
    dz_ok = np.sqrt(diagonal-alpha)

    print('Ordinary Kriging Done')
    #get rid of resulting double brackets
    a = np.squeeze(np.asarray(z_obs_sk))
    b = np.squeeze(np.asarray(dz_sk))
    c = np.squeeze(np.asarray(z_obs_ok))
    d = np.squeeze(np.asarray(dz_ok))
    return a, b, c, d













def kriging_main(covariance, ds_masked, cond_df, flattened_idx, obs_cov, W):
    #obs_cov removed as argument for now
    obs_sk_2d, dz_sk_2d, obs_ok_2d, dz_ok_2d = kriged_output(covariance, cond_df, ds_masked, flattened_idx, obs_cov, W)
    
    """
    obs = cond_df['sst_anomaly'].values #cond_df['cci_anomalies'].values
    print('1 - DONE')
    water_mask, water_idx = watermask(ds_masked, 0)
    obs_idx = flattened_idx
    unique_obs_idx = np.unique(obs_idx)
    print('2 - DONE')
    obs_sk, dz_sk, obs_ok, dz_ok = krige_for_esa_values_only(water_idx, unique_obs_idx, W, obs, covariance)
    print('3 - DONE')
    obs_sk_2d = result_reshape_2d(obs_sk, water_idx, water_mask)
    print('4 - DONE')
    dz_sk_2d = result_reshape_2d(dz_sk, water_idx, water_mask)
    print('5 - DONE')
    obs_ok_2d = result_reshape_2d(obs_ok, water_idx, water_mask)
    print('6 - DONE')
    dz_ok_2d = result_reshape_2d(dz_ok, water_idx, water_mask)
    print('7 - DONE')
    """
    
    return obs_sk_2d, dz_sk_2d, obs_ok_2d, dz_ok_2d
