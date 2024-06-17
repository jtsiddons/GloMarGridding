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


#math tools 
import numpy as np

#data handling tools
import pandas as pd
import polars as pl
import xarray as xr

#time tools
from datetime import datetime
import calendar

#from pyarrow.lib import ArrowInvalid




def is_single_item_list(list_to_check):
    #Check that list is not empty
    try:
        _ = list_to_check[0]
    except IndexError:
        return False
    #Return true if list has a single element
    try:
        _ = list_to_check[1]
    except IndexError:
        return True
    #Return False if more than one element
    return False




def read_in_data(data_path, year=False, month=False,  obs=False, subdirectories=False):
    ds_dir = [x[0] for x in os.walk(data_path)] #os.walk(path)
    print(ds_dir)
    if obs is True and subdirectories is False:
        print('there are files')
        ds_dir = (ds_dir[0])
        #print(ds_dir)
        
        long_filelist = []
        filelist = sorted(os.listdir(ds_dir)) #_fullpath(dirname)
        print(filelist)
        
        #r = re.compile(str(year)+'_'+str(month).zfill(2) + '.csv')
        #filtered_list = list(filter(r.match, fdef rmse(predictions, targets):
        
        
        #for multiple months 
        #(when processing MetOffice pentads, there might be need for a few days from before/after main month
        mon_list = [month-1, month, month+1]
        str_list = [str(year)+'_'+str(i).zfill(2)+'.csv' for i in mon_list]
        print(mon_list)
        print(str_list)
        filtered_list = [i for i in filelist if i in str_list]
        print(filtered_list)
        
        fullpath_list = [os.path.join(ds_dir,f) for f in filtered_list]
        long_filelist.extend(fullpath_list)
        print(long_filelist)
        
    elif obs is False and subdirectories is False:
        print('there are files')
        ds_dir = (ds_dir[0])
        #print(ds_dir)
        
        long_filelist = []
        filelist = sorted(os.listdir(ds_dir)) #_fullpath(dirname)
        print(filelist)
        #r = re.compile(str(year)+'_'+str(month).zfill(2) + '.feather')
        #filtered_list = list(filter(r.match, filelist))
        
        #for multiple months 
        #(when processing MetOffice pentads, there might be need for a few days from before/after main month
        mon_list = [month-1, month, month+1]
        str_list = [str(year)+'_'+str(i).zfill(2)+'.feather' for i in mon_list]
        print(mon_list)
        print(str_list)
        filtered_list = [i for i in filelist if i in str_list]
        print(filtered_list)
        
        fullpath_list = [os.path.join(ds_dir,f) for f in filtered_list]
        long_filelist.extend(fullpath_list)
        print(long_filelist)
        
    else:
        print('there are subdirectories')
        ds_dir = (ds_dir[1:])
        #print(ds_dir)
    
        long_filelist = []
        for dirname in sorted(ds_dir):
            filelist = sorted(os.listdir(dirname)) #_fullpath(dirname))
            print(filelist)
            #r = re.compile(str(year)+'_'+str(month).zfill(2) + '.feather')
            #filtered_list = list(filter(r.match, filelist))
            #print(filtered_list)
            
            #for multiple months 
            #(when processing MetOffice pentads, there might be need for a few days from before/after main month
            mon_list = [month-1, month, month+1]
            str_list = [str(year)+'_'+str(i).zfill(2)+'.feather' for i in mon_list]
            print(mon_list)
            print(str_list)
            filtered_list = [i for i in filelist if i in str_list]
            print(filtered_list)
            
            fullpath_list = [os.path.join(dirname,f) for f in filtered_list]
            long_filelist.extend(fullpath_list)
        print(long_filelist)
    return long_filelist





def main(data_path, qc_path, qc_path_2, year, month):
    data_dir = read_in_data(data_path, year=year, month=month, obs=True)
    qc_dir_1 = read_in_data(qc_path, year=year, month=month, subdirectories=True)
    qc_dir_2 = read_in_data(qc_path_2, year=year, month=month)
    qc_dir =  qc_dir_1 + qc_dir_2
    #qc_dir = qc_dir[:-1]
    print(qc_dir_1)
    print(qc_dir_2)
    print(qc_dir)
    
    # test somewhere here whether there are any files from the QC dir
    # do the same for obs
    # if no obs or no QC - set as empty and proceed straight to creating an empty netcdf layer
    # if obs and QC both present - proceed with processing as normal
    data_df = pd.read_csv(data_dir[0])
    
    qc_df = pd.DataFrame() #create the empty dataframe
    key_columns = ['any_flag', 'point_dup_flag', 'track_dup_flag']
    print(qc_dir)
    
    data_columns = ['yr', 'mo', 'dy', 'hr', 'lat', 'lon', 'sst', 'ii', 'id', 'orig_id', 'uid', 'dck','data_type']
    qc_columns = ['noval_sst', 'freez_sst', 'hardlim_sst', 'nonorm_sst', 'clim_sst', 'any_flag', 'point_dup_flag', 'track_dup_flag']
    columns_wanted = data_columns + qc_columns
    
    for i in range (0,len(qc_dir),1):
        if any(k not in pl.read_ipc_schema(qc_dir[i]) for k in key_columns):
            continue
        qc_df_i = pd.read_feather(qc_dir[i], columns=columns_wanted)
        #buddy check flag to add
        #qc_df_i = pd.read_feather(qc_dir[i], columns=['uid', 'dck', 'noval_sst', 'freez_sst', 'hardlim_sst', 'nonorm_sst', 'clim_sst', 'any_flag', 'point_dup_flag', 'track_dup_flag'])
        print(qc_df_i.columns.values)
        
        qc_df = pd.concat([qc_df, qc_df_i])
        del qc_df_i
        print('QC DF', qc_df)
        
    if qc_df.shape[0] == 0:
        raise ValueError("No data, or don't have the flags")
    else:
        #qc_df = qc_df[(qc_df['dck']<1000) & 
        qc_df = qc_df[~qc_df[qc_columns].any(axis=1)]
        print(qc_df)
        
        #when merging with FINAL_PROC datafiles, which are not pre-appended
        #qc_df['uid'] = qc_df['uid'].str.slice(-6)
        
        #extra bit to check for duplicates in uid
        duplicate_values = qc_df['uid'].duplicated()
        #print(duplicate_values[duplicate_values== True])
        # remove duplicate values in uid column
        qc_df = qc_df.drop_duplicates(subset=['uid'], keep='last')
        qc_df.drop(columns='dck')
        qc_df = qc_df[(qc_df['any_flag'] == False) & (qc_df['point_dup_flag'] <= 1) & (qc_df['track_dup_flag'] == False)]
        print('QC DF', qc_df) #.columns)
        
    
    # v-stack the qc_df here
    #qc_df_merged = pd.concat(qc_df)

    #add a loop over qc_df files to match with the data_df
    #joined_df = data_df.merge(qc_df, how='inner', on='uid')
    joined_df = qc_df
    #extra bit 
    #create a date (to subsample into MetOffice pentads)
    joined_df['date'] = pd.to_datetime(dict(year=joined_df.yr, month=joined_df.mo, day=joined_df.dy))
    print(joined_df)
    return(joined_df)
    

def MAT_observations(obs_path, obs_path_2, year, month):
    #obs_path is to Joe's data directories
    obs_dir_1 = read_in_data(obs_path, year=year, month=month, subdirectories=True)
    obs_dir_2 = read_in_data(obs_path_2, year=year, month=month)
    obs_dir =  obs_dir_1 + obs_dir_2
    
    print(obs_dir_1)
    print(obs_dir_2)
    print(obs_dir)
    
    obs_df = pd.DataFrame()
    
    data_columns = ['yr', 'mo', 'dy', 'hr', 'datetime', 'local_datetime', 'lat', 'lon', 'at', 'ii', 'id', 'orig_id', 'uid', 'dck','data_type']
    qc_columns = ['any_flag', 'point_dup_flag', 'track_dup_flag']
    columns_wanted = data_columns + qc_columns
    
    for i in range (0,len(obs_dir),1):
        if any(k not in pl.read_ipc_schema(obs_dir[i]) for k in qc_columns):
            continue
        obs_df_i = pd.read_feather(obs_dir[i], columns=columns_wanted)
        print(obs_df_i.columns.values)
        
        obs_df = pd.concat([obs_df, obs_df_i])
        del obs_df_i
        print('OBS DF', obs_df)

    if obs_df.shape[0] == 0:
        raise ValueError("No data, or don't have the flags")
    else:
        obs_df = obs_df[~obs_df[qc_columns].any(axis=1)]
        print(obs_df)
        
    return obs_df


def MAT_heigh_adj(height_path, year, height_member):
    #height_path is to Richard's gzip files with 200 members
    ds_dir = [x[0] for x in os.walk(height_path)][0] #os.walk(path)
    print(ds_dir)
    filelist = sorted(os.listdir(ds_dir)) #_fullpath(dirname)
    print(filelist)
    chosen_filename = ['MAT_hgt_' + str(year)+'_t10m.csv.gz']
    filtered_list = [i for i in filelist if i in chosen_filename]
    print(filtered_list)
    height_file = [os.path.join(ds_dir,f) for f in filtered_list][0]
    print(height_file)
    height_df = pd.read_csv(height_file)
    print(height_df)
    columns = ['uid', 'end '+str(height_member)]
    print(columns)
    height_df = height_df[columns]
    print(height_df)
    return height_df


def MAT_qc(qc_path, year, month):
    #qc_path is to Richard's qc files
    ds_dir = [x[0] for x in os.walk(qc_path)][0] #os.walk(path)
    print(ds_dir)
    filelist = sorted(os.listdir(ds_dir)) #_fullpath(dirname)
    print(filelist)
    chosen_filename = ['MAT_QC_' + str(year)+'.csv']
    filtered_list = [i for i in filelist if i in chosen_filename]
    print(filtered_list)
    fullpath_list = [os.path.join(ds_dir,f) for f in filtered_list]
    print(fullpath_list)
    qc_dir = fullpath_list[0]
    print(qc_dir)
    #data_df = pd.read_csv(fullpath_list[0])
    
    qc_df = pd.read_csv(qc_dir)
    print(qc_df.columns.to_list())
    qc_columns = ['buddy', 'clim_mat', 'hardlimit_mat', 'blklst', 'qc_mat_blacklist']
    data_columns = ['uid']
    needed_columns = qc_columns + data_columns
    print(needed_columns)
    qc_df = qc_df[needed_columns]
    print(qc_df)
    qc_df = qc_df[(qc_df['buddy'] == 0 ) & (qc_df['clim_mat'] == 0) & (qc_df['hardlimit_mat'] == 0) & (qc_df['blklst'] == 0) & (qc_df['qc_mat_blacklist'] == 0)]
    print(qc_df)
    return qc_df


#def MAT_main(obs_path, obs_path_2, height_path, qc_path, year, month, height_member):
def MAT_main(obs_path, obs_path_2, qc_path, year, month):
    obs_df = MAT_observations(obs_path, obs_path_2, year, month)
    qc_df = MAT_qc(qc_path, year, month)
    #height_df = MAT_heigh_adj(height_path, year, height_member)
    
    #when merging with qc and height datafiles, which are not pre-appended
    obs_df['uid'] = obs_df['uid'].str.slice(-6)
    #merge obs_df with qc_df and height adjustments using UID
    joined_df = obs_df.merge(qc_df, how='inner', on='uid')
    #height_adjusted_df = joined_df.merge(height_df, how='inner', on='uid')
    #print(height_adjusted_df)
    print(obs_df['uid'])
    print(qc_df['uid'])
    print('Obs files from Joe')
    print(len(obs_df))
    print('QC files from Richard')
    print(len(qc_df))
    print('Joined df')
    print(len(joined_df))
    #print('Height adjustment df')
    #print(len(height_df))
    #print('Joined df with height adjustemnt merged on')
    #print(len(height_adjusted_df))
    return joined_df #height_adjusted_df


def MAT_add_height_adjustment(joined_df, height_path, year, height_member):
    height_df = MAT_heigh_adj(height_path, year, height_member)
    height_adjusted_df = joined_df.merge(height_df, how='inner', on='uid')
    print(height_adjusted_df)
    height_adjusted_df.columns = [*height_adjusted_df.columns[:-1], 'height']
    # Change column height's values to floats
    height_adjusted_df['height'] = height_adjusted_df['height'].astype(float)
    print(height_adjusted_df)
    print(height_adjusted_df.dtypes)
    height_adjusted_df['obs_anomalies_height'] = height_adjusted_df['obs_anomalies'] - height_adjusted_df['height']
    print(height_adjusted_df)
    return height_adjusted_df


def MAT_match_climatology(df, climatology):
    obs_lat = df['lat']
    obs_lon = df['lon']
    df['DOY'] = df['datetime'].dt.dayofyear
    print(df)
    obs_DOY = df['DOY']
    df['is_leap'] = df['yr'].apply(calendar.isleap)
    print(df)
    if df.loc[df['is_leap'] == True]:
        df['DOY_nonleap'] = df['DOY'] - 1
        print(df)
    elif df.loc[df['is_leap'] == False]:
        print(df)
    return df


def MAT_match_climatology_to_obs(climatology_365, obs_df):
    obs_lat = obs_df.lat
    obs_lon = obs_df.lon
    print(obs_lat)
    print(obs_lon)
    obs_df['lat_idx'], obs_df['lon_idx'] = find_latlon_idx(climatology_365, obs_lat, obs_lon)
    cci_clim = [] #ESA CCI climatology values                                                             
    
    mask = (obs_df['datetime'].dt.is_leap_year == 1) & (obs_df['datetime'].dt.dayofyear == 60)
    #print(obs_df['date'])
    #print(mask)
    non_leap_df = obs_df[~mask]
    leap_df = obs_df[mask]
    
    print(obs_df)
    print('-------------------------------')
    print(non_leap_df)
    print('-------------------------------')
    print(leap_df)
    print('-------------------------------')
    
    non_leap_df['fake_non_leap_year'] = 2010
    non_leap_df['fake_non_leap_date'] = pd.to_datetime(dict(year=non_leap_df.fake_non_leap_year, month=non_leap_df.mo, day=non_leap_df.dy))
    non_leap_df['doy'] = [int(i.strftime('%j')) for i in non_leap_df['fake_non_leap_date']]
    print(non_leap_df.doy)
    non_leap_df['doy_idx'] = non_leap_df.doy - 1
    print(non_leap_df.doy_idx)
    
    print(climatology_365)
    c = climatology_365.values
    #print(c.shape)
    
    selected = c[np.array(non_leap_df.doy_idx), np.array(non_leap_df.lat_idx), np.array(non_leap_df.lon_idx)]
    print(selected)
    #selected = selected - 273.15 #from Kelvin to Celsius
    #print(selected)
    
    #print(len(non_leap_df.lat_idx), len(non_leap_df.lon_idx), len(non_leap_df.doy_idx), len(selected))
    
    #climatology_365.sel(lat=obs_lat, lon=obs_lon, method="nearest") #climatology_365.lat.values[lat_idx]
    end_feb = c[np.repeat(np.array([58]), len(leap_df.lat_idx)), np.array(leap_df.lat_idx), np.array(leap_df.lon_idx)]
    #climatology_365[tuple((leap_df.lat_idx, leap_df.lon_idx, np.repeat(np.array([58]), len(leap_df.lat_idx))))] - cannot use tuple!
    
    beg_mar = c[np.repeat(np.array([59]), len(leap_df.lat_idx)), np.array(leap_df.lat_idx), np.array(leap_df.lon_idx)]
    #climatology_365[tuple((leap_df.lat_idx, leap_df.lon_idx, np.repeat(np.array([59]), len(leap_df.lat_idx))))] - cannot use tuple!
    
    selected2 = [(g + h) / 2 for g, h in zip(end_feb, beg_mar)]
    #selected2 = [ i - 273.15 for i in selected2] #from Kelvin to Celsius
    #print(selected2)
    
    
    non_leap_df['climatology'] = selected
    leap_df['climatology'] = selected2
    obs_df = pd.concat([non_leap_df, leap_df])
    #print('joint leap and non-leap observations', obs_df)
    obs_df['obs_anomalies'] = obs_df['at'] - obs_df['climatology']
    #df1 = obs_df[['lat', 'lon', 'sst', 'climatology', 'cci_anomalies', 'obs_anomalies']]
    #print(df1)
    #STOP
    """
    #in case some of the values are Nan (because covered by ice)                                                                     
    obs_df = obs_df.dropna()
    """
    print(obs_df)
    return obs_df


def SST_match_climatology_to_obs(climatology_365, obs_df):
    obs_lat = obs_df.lat
    obs_lon = obs_df.lon
    print(obs_lat)
    print(obs_lon)
    obs_df['lat_idx'], obs_df['lon_idx'] = find_latlon_idx(climatology_365, obs_lat, obs_lon)
    cci_clim = [] #ESA CCI climatology values                                                             
    
    mask = (obs_df['date'].dt.is_leap_year == 1) & (obs_df['date'].dt.dayofyear == 60)
    #print(obs_df['date'])
    #print(mask)
    non_leap_df = obs_df[~mask]
    leap_df = obs_df[mask]
    print(obs_df)
    print('-------------------------------')
    print(non_leap_df)
    print('-------------------------------')
    print(leap_df)
    print('-------------------------------')
    non_leap_df['fake_non_leap_year'] = 2010
    non_leap_df['fake_non_leap_date'] = pd.to_datetime(dict(year=non_leap_df.fake_non_leap_year, month=non_leap_df.mo, day=non_leap_df.dy))
    non_leap_df['doy'] = [int(i.strftime('%j')) for i in non_leap_df['fake_non_leap_date']]
    print(non_leap_df.doy)
    non_leap_df['doy_idx'] = non_leap_df.doy - 1
    print(non_leap_df.doy_idx)
    
    print(climatology_365)
    c = climatology_365.values
    #print(c.shape)
    selected = c[np.array(non_leap_df.doy_idx), np.array(non_leap_df.lat_idx), np.array(non_leap_df.lon_idx)]
    print(selected)
    
    selected = selected - 273.15 #from Kelvin to Celsius
    #print(selected)
    
    #print(len(non_leap_df.lat_idx), len(non_leap_df.lon_idx), len(non_leap_df.doy_idx), len(selected))
    
    #climatology_365.sel(lat=obs_lat, lon=obs_lon, method="nearest") #climatology_365.lat.values[lat_idx]
    end_feb = c[np.repeat(np.array([58]), len(leap_df.lat_idx)), np.array(leap_df.lat_idx), np.array(leap_df.lon_idx)]
    #climatology_365[tuple((leap_df.lat_idx, leap_df.lon_idx, np.repeat(np.array([58]), len(leap_df.lat_idx))))] - cannot use tuple!
    
    beg_mar = c[np.repeat(np.array([59]), len(leap_df.lat_idx)), np.array(leap_df.lat_idx), np.array(leap_df.lon_idx)]
    #climatology_365[tuple((leap_df.lat_idx, leap_df.lon_idx, np.repeat(np.array([59]), len(leap_df.lat_idx))))] - cannot use tuple!
    
    selected2 = [(g + h) / 2 for g, h in zip(end_feb, beg_mar)]
    selected2 = [ i - 273.15 for i in selected2] #from Kelvin to Celsius
    #print(selected2)
    
    
    non_leap_df['climatology'] = selected
    leap_df['climatology'] = selected2
    obs_df = pd.concat([non_leap_df, leap_df])
    #print('joint leap and non-leap observations', obs_df)
    obs_df['sst_anomaly'] = obs_df['sst'] - obs_df['climatology']
    #df1 = obs_df[['lat', 'lon', 'sst', 'climatology', 'cci_anomalies', 'obs_anomalies']]
    #print(df1)
    #STOP
    """
    #in case some of the values are Nan (because covered by ice)                                                                     
    obs_df = obs_df.dropna()
    """
    print(obs_df)
    return obs_df


def find_latlon_idx(nc_xr, lat, lon):
    '''
    Parameters:
    cci (array) - array of cci  vaules for each point in the whole domain
    lat (array) - array of latitudes of the observations
    lon (array) - array of longitudes of the observations
    df (dataframe) - dataframe containing all information for a given day (such as location, measurement values)

    Returns:
    Dataframe with added anomalies for each observation point
    '''
    #print(nc_xr)
    lat_idx = find_nearest(nc_xr.lat, lat)  
    #print(nc_xr.lat, lat_idx)
    lon_idx = find_nearest(nc_xr.lon, lon)  
    #print(nc_xr.lon, lon_idx)
    return lat_idx, lon_idx


def find_nearest(array, values):
    array = np.asarray(array)
    idx_list = [(np.abs(array - value)).argmin() for value in values]
    return idx_list
