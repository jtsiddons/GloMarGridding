"""Functions for reading and loading covariance matrices from NetCDF files"""
#
# by A. Faulkner
# for python version 3.0 and up
#

# global
import os
import re
from warnings import warn

import numpy as np

# data handling tools
import xarray as xr
from xarray.backends.api import T_Engine

from .utils import MonthName, regex_coord


def _preprocess(x, lon_bnds, lat_bnds):
    return x.sel(lon=slice(*lon_bnds), lat=slice(*lat_bnds))


# WARN: Specific function - maybe pass a list of patterns?
def read_in_covariance_file(
    path: str,
    month: int,
) -> xr.Dataset:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path {path} not found")

    if os.path.isfile(path):
        return xr.open_dataset(path, engine="netcdf4")

    # Is a directory
    filelist: list[str] = os.listdir(path)
    if not filelist:
        raise FileNotFoundError(f"Directory {path} is empty")

    mon_str: str = MonthName(month).name.lower()
    print(f"Matching global covariance for {mon_str}")

    r: re.Pattern = re.compile(f"world_{mon_str}" + r"\w+.nc")
    filtered_list: list[str] = [f for f in filelist if r.match(f)]
    if not filtered_list:
        r: re.Pattern = re.compile(f"covariance_{month:02d}" + r"\w+.nc")
        filtered_list: list[str] = [f for f in filelist if r.match(f)]
    if not filtered_list:
        raise FileNotFoundError(f"Cannot find monthly file in directory {path}")

    if len(filtered_list) > 1:
        warn(
            f"Found multiple files in {path}. Taking first. All files: "
            + ", ".join(filtered_list)
        )

    return xr.open_dataset(
        os.path.join(path, filtered_list[0]), engine="netcdf4"
    )


def get_covariance(
    path: str,
    month: int,
    cov_var_name: str = "covariance",
) -> np.ndarray:
    ds = read_in_covariance_file(path, month)
    return ds.variables[cov_var_name].values


def read_single_file(
    path: str,
    engine: T_Engine,
    adjust_lon: bool = True,
    rename_lon: str | None = None,
    rename_lat: str | None = None,
) -> tuple[xr.Dataset, str, str]:
    ds = xr.open_dataset(path, engine=engine)

    lon_var_name = regex_coord(
        ds, re.compile(r"^lon(gitude)?$"), case_insensitive=True
    )
    if rename_lon:
        ds.rename({lon_var_name: rename_lon})
        lon_var_name = rename_lon

    lat_var_name = regex_coord(
        ds, re.compile(r"^lat(itude)?$"), case_insensitive=True
    )
    if rename_lat:
        ds.rename({lat_var_name: rename_lat})
        lat_var_name = rename_lat

    if adjust_lon and (ds.coords[lon_var_name] > 180).any():
        ds.coords[lon_var_name] = (
            (ds.coords[lon_var_name] + 540.0) % 360.0
        ) - 180.0
    return ds, lon_var_name, lat_var_name


# def get_singlefile_landmask(path: str):
#     ds = xr.open_dataset(str(path), engine="netcdf4")
#     print(ds)
#     try:
#         landmask = ds.variables["landice_sea_mask"].values
#     except KeyError:
#         landmask = ds.variables["landmask"].values
#     print(landmask)
#     lat = ds.lat.values
#     lon = ds.lon.values
#     print(lon)
#     lon = ((lon + 540.0) % 360.0) - 180.0
#     print(lon)
#     # water is 1, land is 0
#     ds.coords["lon"] = lon
#     print(ds.lat.values)
#     print(ds.lon.values)
#     # gridlon, gridlat = np.meshgrid(esa_cci_lon, esa_cci_lat)
#     return ds, lat, lon


"""
if __name__ == "__main__":
    covariance_main(path, ds_varname, Wlon, Elon, Slat, Nlat, cov_choice, ipc=False, time_resolution=False)
"""
