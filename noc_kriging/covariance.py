#
# by A. Faulkner
# for python version 3.0 and up
#

# global
import os
import re
import os.path

# argument parser
from configparser import ConfigParser

# data handling tools
import xarray as xr


def _preprocess(x, lon_bnds, lat_bnds):
    return x.sel(lon=slice(*lon_bnds), lat=slice(*lat_bnds))


def read_in_covariance_file(path, month):
    # for a path to a directory with covariances
    if os.path.isdir(path):
        monthDict = {
            1: "january",
            2: "february",
            3: "march",
            4: "april",
            5: "may",
            6: "june",
            7: "july",
            8: "august",
            9: "september",
            10: "october",
            11: "november",
            12: "december",
        }
        long_filelist = []

        filelist = os.listdir(path)  # os.path.join(thedirectory, thefile)
        print(filelist)

        mon_str = monthDict[int(month)]
        print("Matching global covariance for %s" % mon_str)
        r = re.compile("world_" + str(mon_str) + r"\w+.nc")
        filtered_list = list(filter(r.match, filelist))
        if not filtered_list:
            r = re.compile("covariance_" + str(month).zfill(2) + r"\w+.nc")
        filtered_list = list(filter(r.match, filelist))
        fullpath_list = [os.path.join(path, f) for f in filtered_list]
        print(filtered_list)
        print(fullpath_list)
        # long_filelist.extend(fullpath_list)
        # print(long_filelist)

        ds = xr.open_dataset(fullpath_list[0], engine="netcdf4")
        print(ds)
    # for a path to a single covariance file
    elif os.path.isfile(path):
        # for a single file covariance
        ds = xr.open_dataset(str(path), engine="netcdf4")
    print(ds)
    return ds


def get_covariance(path, month):
    ds = read_in_covariance_file(path, month)
    covariance = ds.variables["covariance"].values
    print(covariance)
    return covariance


def get_landmask(path, month):
    ds = read_in_covariance_file(path, month)
    landmask = ds.variables["landice_sea_mask"].values
    print(landmask)
    lat = ds.latitude.values
    lon = ds.longitude.values
    print(lon)
    lon = ((lon + 540.0) % 360.0) - 180.0
    print(lon)
    # water is 1, land is 0
    ds.coords["longitude"] = lon
    print(ds.latitude.values)
    print(ds.longitude.values)
    # gridlon, gridlat = np.meshgrid(esa_cci_lon, esa_cci_lat)
    return ds, lat, lon


def get_singlefile_landmask(path):
    ds = xr.open_dataset(str(path), engine="netcdf4")
    print(ds)
    try:
        landmask = ds.variables["landice_sea_mask"].values
    except KeyError:
        landmask = ds.variables["landmask"].values
    print(landmask)
    lat = ds.lat.values
    lon = ds.lon.values
    print(lon)
    lon = ((lon + 540.0) % 360.0) - 180.0
    print(lon)
    # water is 1, land is 0
    ds.coords["lon"] = lon
    print(ds.lat.values)
    print(ds.lon.values)
    # gridlon, gridlat = np.meshgrid(esa_cci_lon, esa_cci_lat)
    return ds, lat, lon


"""
if __name__ == "__main__":
    covariance_main(path, ds_varname, Wlon, Elon, Slat, Nlat, cov_choice, ipc=False, time_resolution=False)
"""
