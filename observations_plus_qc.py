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


#math tools 
import numpy as np

#data handling tools
import pandas as pd
import xarray as xr





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




def read_in_data(data_path, year=False, month=False,  subdirectories=False):
    ds_dir = [x[0] for x in os.walk(data_path)] #os.walk(path)
    
    if subdirectories is False:
        print('there are files')
        ds_dir = (ds_dir[0])
        #print(ds_dir)
        
        long_filelist = []
        filelist = sorted(os.listdir(ds_dir)) #_fullpath(dirname)
        #print(filelist)
        r = re.compile(str(year)+'_'+str(month).zfill(2) + '.csv')
        filtered_list = list(filter(r.match, filelist))
        fullpath_list = [os.path.join(ds_dir,f) for f in filtered_list]
        long_filelist.extend(fullpath_list)
        #print(long_filelist)
    
    else:
        print('there are subdirectories')
        ds_dir = (ds_dir[1:])
        #print(ds_dir)
    
        long_filelist = []
        for dirname in sorted(ds_dir):
            filelist = sorted(os.listdir(dirname)) #_fullpath(dirname))
            #print(filelist)
            r = re.compile(str(year)+'_'+str(month).zfill(2) + '.feather')
            filtered_list = list(filter(r.match, filelist))
            fullpath_list = [os.path.join(dirname,f) for f in filtered_list]
            long_filelist.extend(fullpath_list)
        #print(long_filelist)
    return long_filelist





def main(data_path, qc_path, year, month):
    data_dir = read_in_data(data_path, year=year, month=month)
    qc_dir = read_in_data(qc_path, year=year, month=month, subdirectories=True)
    qc_dir = qc_dir[:-1]
    
    data_df = pd.read_csv(data_dir[0])
    
    qc_df = [] #create the empty dataframe
    for i in range (0,len(qc_dir),1):
        qc_df_i = pd.read_feather(qc_dir[i], columns=['uid', 'dck', 'noval_sst', 'freez_sst', 'hardlim_sst', 'nonorm_sst', 'clim_sst'])
        #print(qc_df_i.columns.values)
        
        qc_cols = qc_df_i.columns.values[2:]
        qc_df_i = qc_df_i[(qc_df_i['dck']<1000) & ~(qc_df_i[qc_cols].any(axis=1))]
        qc_df_i['uid'] = qc_df_i['uid'].str.slice(-6)
        #extra bit to check for duplicates in uid
        duplicate_values = qc_df_i['uid'].duplicated()
        #print(duplicate_values[duplicate_values== True])
        # remove duplicate values in uid column
        qc_df_i = qc_df_i.drop_duplicates(subset=['uid'], keep='first')
        ###
        qc_df_i.drop(columns='dck')
        # append the dfs that have data
        if not qc_df_i.empty:
            qc_df.append(qc_df_i)
            #print(qc_df_i)
    
    # v-stack the qc_df here
    qc_df_merged = pd.concat(qc_df)
    #add a loop over qc_df files to match with the data_df
    joined_df = data_df.merge(qc_df_merged, how='inner', on='uid')
    print(joined_df)
    return(joined_df)
"""
#TO DO - DEVELOP FOR LOOPING OVER EACH YEAR AND MONTH FROM 1900
for year in range (1900,2014,1):
    for month in range (1,13,1):
        data_dir = read_in_data(data_path, year, month)
        qc_dir = read_in_data(qc_path, year, month, subdirectories=True)
"""

"""
parser = argparse.ArgumentParser()
parser.add_argument("-config", dest="config", required=False, default="config.ini", help="INI file containing configuration settings")
args = parser.parse_args()
config_file = args.config
print(config_file)

#instantiate
config = ConfigParser()
#parce existing config file
config.read(config_file) #('config.ini' or 'three_step_kriging.ini')

data_path = config.get('observations', 'observations')
qc_path = config.get('observations', 'qc_flags_joe')

main(data_path, qc_path, year=2000, month=1)
"""
