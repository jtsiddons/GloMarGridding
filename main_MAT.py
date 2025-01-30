#!/usr/bin/env python

################
# by A. Faulkner
# for python version 3.0 and up
################

# global
import os

# argument parser
import argparse

import yaml

# math tools
import numpy as np

# timing tools
from calendar import isleap

# import datetime as dt
from datetime import date, timedelta
from netCDF4 import date2num

# data handling tools
import netCDF4 as nc
import pandas as pd
import polars as pl
import xarray as xr

# self-written modules (from the same directory)
from glomar_gridding.distances import haversine_gaussian
from glomar_gridding.mask import mask_observations
from glomar_gridding.climatology import match_climatology
from glomar_gridding.interpolation_covariance import load_covariance
from glomar_gridding.utils import get_pentad_range
import glomar_gridding.observations as obs_module
import glomar_gridding.observations_plus_qc as obs_qc_module
import glomar_gridding.kriging as krig
import glomar_gridding.error_covariance as err_cov

from .noc_helpers import add_height_adjustment

# PyCOADS functions
from PyCOADS.processing.solar import is_daytime

parser = argparse.ArgumentParser()
parser.add_argument(
    "-config",
    dest="config",
    required=False,
    default=os.path.join(__file__, "config_mat.yaml"),
    help="INI file containing configuration settings",
)
parser.add_argument(
    "-year_start",
    dest="year_start",
    required=False,
    help="start year",
    type=int,
)
parser.add_argument(
    "-year_stop",
    dest="year_stop",
    required=False,
    help="end year",
    type=int,
)
parser.add_argument(
    "-height_member",
    dest="height_member",
    required=False,
    help="height member: if height member is 0, no height adjustment is performed.",
    type=int,
    default=0,
)
parser.add_argument(
    "-method",
    dest="method",
    default="simple",
    required=False,
    help='Kriging Method - one of "simple" or "ordinary"',
)


def _parse_args(parser) -> tuple[dict, int, int, int, str]:
    args = parser.parse_args()
    config: dict = yaml.safe_load(args.config)
    year_start: int = args.year_start or config.get("domain", {}).get(
        "startyear", 1900
    )
    year_stop: int = args.year_stop or config.get("domain", {}).get(
        "endyear", 2024
    )

    return config, year_start, year_stop, args.height_member, args.method


def _initialise_ncfile(
    ncfile: nc.Dataset,
    output_lon: np.ndarray,
    output_lat: np.ndarray,
    current_year: int,
    height_member: int = 0,
    adjusted_height: float | None = None,
):
    ncfile.createDimension("lat", len(output_lat))  # latitude axis
    ncfile.createDimension("lon", len(output_lon))  # longitude axis
    ncfile.createDimension("time", None)  # unlimited axis

    # Define two variables with the same names as dimensions,
    # a conventional way to define "coordinate variables".
    lat = ncfile.createVariable("lat", np.float32, ("lat",))
    lat.units = "degrees_north"
    lat.long_name = "latitude"
    lat[:] = output_lat  # ds.lat.values
    lon = ncfile.createVariable("lon", np.float32, ("lon",))
    lon.units = "degrees_east"
    lon.long_name = "longitude"
    lon[:] = output_lon  # ds.lon.values
    time = ncfile.createVariable("time", np.float32, ("time",))
    time.units = "days since %s-01-01" % (str(current_year))
    time.long_name = "time"
    # print(time)

    # Define a 3D variable to hold the data
    if height_member:
        krig_anom = ncfile.createVariable(
            f"mat_anomaly_{adjusted_height}m",
            np.float32,
            ("time", "lat", "lon"),
        )
        # note: unlimited dimension is leftmost
        krig_anom.standard_name = f"MAT anomaly at {adjusted_height} m"
        krig_anom.height = str(adjusted_height)
        krig_anom.ensemble_member = height_member
    else:
        krig_anom = ncfile.createVariable(
            "mat_anomaly", np.float32, ("time", "lat", "lon")
        )
        krig_anom.standard_name = "MAT anomaly"
    krig_anom.units = "deg C"  # degrees Kelvin

    # Define a 3D variable to hold the data
    krig_uncert = ncfile.createVariable(
        "mat_anomaly_uncertainty", np.float32, ("time", "lat", "lon")
    )  # note: unlimited dimension is leftmost
    krig_uncert.units = "deg C"  # degrees Kelvin
    krig_uncert.standard_name = "uncertainty"  # this is a CF standard name
    # Define a 3D variable to hold the data
    grid_obs = ncfile.createVariable(
        "observations_per_gridcell", np.float32, ("time", "lat", "lon")
    )
    # note: unlimited dimension is leftmost
    grid_obs.units = ""  # degrees Kelvin
    grid_obs.standard_name = "Number of observations within each gridcell"

    return lon, lat, time, krig_anom, krig_uncert, grid_obs


def _load_landmask(path: str, month: int) -> xr.DataArray:
    return xr.DataArray()


def _load_observations(
    path: str,
    qc_path: str,
    qc_tracked_path: str,
    qc_mat: str,
    year: int,
    month: int,
) -> pl.DataFrame:
    return pl.DataFrame()


def _measurement_covariance(
    df: pl.DataFrame,
    sig_ms: float,
    sig_mb: float,
    sig_bs: float,
    sig_bb: float,
) -> tuple[np.ndarray, np.ndarray]:
    obs_bias_map = {"ship": sig_ms, "buoy": sig_mb}
    covx1 = err_cov.uncorrelated_components(
        df, group_col="data_type", obs_sig_map=obs_bias_map
    )
    dist, weights = err_cov.dist_weight(df, dist_fn=haversine_gaussian)
    covx1 = covx1 + dist
    del dist
    # print(covx1, covx1.shape)
    bias_uncert_map = {"ship": sig_bs, "buoy": sig_bb}
    covx1 = covx1 + err_cov.correlated_components(
        df,
        group_col="data_type",
        bias_sig_map=bias_uncert_map,
    )
    return covx1, weights


def main():  # noqa: D103
    config, year_start, year_stop, height_member, method = _parse_args(parser)

    # location of MAT climatology
    climatology = config.get("climatology", {}).get("path")
    # NOTE: just used for pentad date values for alignment
    metoffice_climatology = config.get("climatology", {}).get(
        "metoffice_climatology"
    )

    # set boundaries for the domain
    lon_west: float = config.get("domain", {}).get("west", -180.0)
    lon_east: float = config.get("domain", {}).get("east", 180.0)
    lat_south: float = config.get("domain", {}).get("south", -90.0)
    lat_north: float = config.get("domain", {}).get("north", 90.0)

    # read measurement and bias uncertainties from config
    sig_ms: float = config.get("parameters", {}).get("sig_ms")
    sig_mb: float = config.get("parameters", {}).get("sig_mb")
    sig_bs: float = config.get("parameters", {}).get("sig_bs")
    sig_bb: float = config.get("parameters", {}).get("sig_bb")

    # location of the ICOADS observation files
    data_dir: str = config.get("observations", {}).get("path")

    height_adjustment_dir: str | None = None
    adjusted_height: int | None = None
    if height_member > 0:
        height_adjustment_dir = config.get("parameters", {}).get(
            "height_adjustments"
        )
        adjusted_height = config.get("parameters", {}).get("adjusted_height")

    # location og QC flags in GROUPS subdirectories
    qc_mat = config.get("observations", {}).get("qc")
    qc_dir = config.get("observations", {}).get("icoads_qc_flags")
    qc_dir_tracked = config.get("observations", {}).get(
        "icoads_qc_flags_tracked"
    )

    # path where the covariance(s) is/are located
    # if single covariance, then full path
    # if several different covariances, then path to directory
    cov_dir: str = config.get("covariance", {}).get("path")
    ellipse_param_dir: str = config.get("covariance", {}).get(
        "ellipse_parameters"
    )
    mask_dir: str = config.get("mask", {}).get("path")

    output_dir: str = config.get("covariance", {}).get("output")

    bnds = [lon_west, lon_east, lat_south, lat_north]
    # extract the latitude and longitude boundaries from user input
    lon_bnds, lat_bnds = (bnds[0], bnds[1]), (bnds[2], bnds[3])
    print(lon_bnds, lat_bnds)

    output_lat = np.arange(lat_bnds[0] + 0.5, lat_bnds[-1] + 0.5, 1)
    output_lon = np.arange(lon_bnds[0] + 0.5, lon_bnds[-1] + 0.5, 1)
    print(output_lat)
    print(output_lon)

    climatology = obs_module.read_climatology(
        climatology, lat_north, lat_south, lon_west, lon_east
    )
    print(climatology)
    clim = climatology.t10m_clim_day
    print(clim)
    del climatology
    # while doing pentad processing, this will set "mid-pentads" dates for the
    # year
    pentad_climatology = obs_module.read_climatology(
        metoffice_climatology, lat_south, lat_north, lon_west, lon_east
    )
    clim_times = pentad_climatology.time
    print(clim_times)
    del pentad_climatology
    # climatology2 = np.broadcast_to(
    #     mask_ds.landmask.values > 0, climatology.climatology.values.shape
    # )

    year_list = range(year_start, year_stop + 1)
    month_list = range(1, 13, 1)

    for current_year in year_list:
        # add MetOffice pentads here
        yr_rng = pl.date_range(
            date(1970, 1, 3), date(1970, 12, 31), interval="5d", eager=True
        ).alias("dates")

        by_month_frame = (
            yr_rng.dt.replace(year=current_year)
            .to_frame()
            .group_by(pl.col("dates").dt.month().alias("month"))
            .agg("dates")
        )
        by_month = {month: dates for month, dates in by_month_frame.rows()}
        del by_month_frame

        try:
            ncfile.close()  # make sure dataset is not already open.
        except (NameError, RuntimeError):
            # ncfile not created yet, or already closed
            pass
        except Exception as e:  # Unknown Error
            raise e

        ncfilename = f"{current_year}_{method}_kriged_MAT"
        if height_member:
            # QUESTION: why not the adjusted height value in the filename?
            ncfilename += f"_heightmember_{height_member:03d}m"
        # if adjusted_height:
        #     ncfilename += f"_height_{adjusted_height:.2f}m"
        ncfilename += ".nc"
        ncfilename = os.path.join(output_dir, ncfilename)

        ncfile = nc.Dataset(ncfilename, mode="w", format="NETCDF4_CLASSIC")
        # print(ncfile)

        _, _, time, krig_anom, krig_uncert, grid_obs = _initialise_ncfile(
            ncfile=ncfile,
            output_lon=output_lon,
            output_lat=output_lat,
            current_year=current_year,
            height_member=height_member,
            adjusted_height=adjusted_height,
        )

        for current_month in month_list:
            # print(current_month)

            monthly = by_month[current_month - 1]

            print(monthly)
            print("Current month and year: ", (current_month, current_year))

            # covariance = cov_module.get_covariance(cov_dir, month=current_month)
            covariance = load_covariance(cov_dir, current_month)
            print(covariance)
            diag_ind = np.diag_indices_from(covariance)
            covariance[diag_ind] = covariance[diag_ind] * 1.01 + 0.005
            print(covariance)

            # WARN: Should this be a landmask file rather than covariance?
            mask_ds = _load_landmask(mask_dir, month=current_month)
            print(mask_ds)

            # read in observations and QC
            obs_df = _load_observations(
                data_dir,
                qc_dir,
                qc_dir_tracked,
                qc_mat,
                year=current_year,
                month=current_month,
            )
            obs_df = obs_df.pipe(is_daytime).filter(pl.col("is_daytime").eq(0))
            print(obs_df)  # [['local_datetime', 'is_daytime']])
            # read in climatology here
            # match with obs against DOY
            print(clim)
            # obs_df = obs_qc_module.MAT_match_climatology(obs_df, clim)
            obs_df: pl.DataFrame = match_climatology(clim, obs_df)

            # merge on the height adjustment
            if (
                height_member > 0
                and adjusted_height is not None
                and height_adjustment_dir is not None
            ):
                obs_df = add_height_adjustment(
                    obs_df,
                    height_adjustment_path=height_adjustment_dir,
                    year=current_year,
                    adjusted_height=adjusted_height,
                    height_member=height_member,
                    mat_col="obs_anomalies",
                )

            print(obs_df)
            print(obs_df.columns)

            # read in ellipse parameters file corresponding to the processed file
            month_ellipse_param = obs_qc_module.ellipse_param(
                ellipse_param_dir, month=current_month, var="MAT"
            )

            # list of dates for each year
            # _, month_range = monthrange(current_year, current_month)
            # print(month_range)

            # if we do MetOffice processing:
            for pentad_idx, pentad_date in enumerate(monthly):
                print(pentad_date)
                print(pentad_idx)

                timestep = pentad_idx
                current_date = pentad_date

                start_date, end_date = get_pentad_range(current_date)
                day_df = obs_df.filter(
                    pl.col("datetime").is_between(
                        start_date, end_date, closed="both"
                    )
                )

                # calculate flattened idx based on the ESA landmask file
                # which is compatible with the ESA-derived covariance
                # mask_ds, mask_ds_lat, mask_ds_lon = obs_module.landmask(water_mask_file, lat_south,lat_north, lon_west,lon_east)

                day_df = mask_observations(
                    day_df,
                    mask_ds,
                    varnames="at",
                )

                # plotting_df = cond_df[['lon', 'lat', 'sst', 'climatology_sst', 'sst_anomaly']]
                # lons = plotting_df['lon']
                # lats = plotting_df['lat']
                # ssts = plotting_df['sst']
                # clims = plotting_df['climatology_sst']
                # anoms = plotting_df['sst_anomaly']
                #
                # skwargs = {'s': 2, 'c': 'red'}
                # fig = plt.figure(figsize=(10, 5))
                # ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
                # cp.projected_scatter(fig, ax, lons, lats, skwargs=skwargs, title='ICOADS locations - '+str(pentad_idx)+' pentad '+ str(current_year)+' year')
                # #plt.show()
                # fig.savefig('/noc/users/agfaul/ellipse_kriging/%s_%spoints.png' % (str(current_year), str(pentad_idx)))
                #
                # skwargs = {'s': 2, 'c': ssts, 'cmap': plt.cm.get_cmap('coolwarm'), 'clim': (-10, 14)}
                # ckwargs = {'label': 'SST [deg C]'}
                # fig = plt.figure(figsize=(10, 5))
                # ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
                # cp.projected_scatter(fig, ax, lons, lats, add_colorbar=True, skwargs=skwargs, ckwargs=ckwargs, title='ICOADS measured SST -' +str(pentad_idx)+ ' pentad ' + str(current_year)+' year', land_col='darkolivegreen')
                # #plt.show()
                # fig.savefig('/noc/users/agfaul/ellipse_kriging/%s_%ssst.png' % (str(current_year), str(pentad_idx)))
                #
                # skwargs = {'s': 2, 'c': clims, 'cmap': plt.cm.get_cmap('coolwarm'), 'clim': (-10, 14)}
                # ckwargs = {'label': 'SST [deg C]'}
                # fig = plt.figure(figsize=(10, 5))
                # ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
                # cp.projected_scatter(fig, ax, lons, lats, add_colorbar=True, skwargs=skwargs, ckwargs=ckwargs, title='ESA CCI climatology - '+str(pentad_idx)+' pentad '+ str(current_year)+' year', land_col='darkolivegreen')
                # #plt.show()
                # fig.savefig('/noc/users/agfaul/ellipse_kriging/%s_%sclim.png' % (str(current_year), str(pentad_idx)))
                #
                # skwargs = {'s': 2, 'c': anoms, 'cmap': plt.cm.get_cmap('coolwarm'), 'clim': (-10, 14)}
                # ckwargs = {'label': 'SST [deg C]'}
                # fig = plt.figure(figsize=(10, 5))
                # ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
                # cp.projected_scatter(fig, ax, lons, lats, add_colorbar=True, skwargs=skwargs, ckwargs=ckwargs, title='ICOADS SST anomalies - '+str(pentad_idx)+ ' pentad '+ str(current_year)+' year', land_col='darkolivegreen')
                # #plt.show()
                # fig.savefig('/noc/users/agfaul/ellipse_kriging/%s_%sanom.png' % (str(current_year), str(pentad_idx)))

                # day_flat_idx = day_df.get_column(["flattened_idx")

                # match gridded observations to ellipse parameters
                day_df = obs_module.match_ellipse_parameters_to_gridded_obs(
                    month_ellipse_param, day_df, mask_ds
                )

                day_df["gridbox"] = day_flat_idx  # .values.reshape(-1)
                gridbox_counts = day_df["gridbox"].value_counts()
                gridbox_count_np = gridbox_counts.to_numpy()
                gridbox_id_np = gridbox_counts.index.to_numpy()
                del gridbox_counts
                water_mask = np.copy(
                    mask_ds.variables["landice_sea_mask"][:, :]
                )
                grid_obs_2d = krig.result_reshape_2d(
                    gridbox_count_np, gridbox_id_np, water_mask
                )

                obs_covariance, W = _measurement_covariance(
                    day_df, sig_ms, sig_mb, sig_bs, sig_bb
                )
                print(obs_covariance)
                print(W)

                # krige obs onto gridded field
                anom, uncert = krig.kriging_main(
                    covariance,
                    day_df,
                    mask_ds,
                    day_flat_idx,
                    obs_covariance,
                    W,
                    kriging_method=method,
                )
                print("Kriging done, saving output")

                # fig = plt.figure(figsize=(10, 5))
                # img_extent = (-180., 180., -90., 90.)
                # ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
                # ax.set_extent([-180., 180., -90., 90.], crs=ccrs.PlateCarree())
                # ax.add_feature(cfeature.LAND, color='darkolivegreen')
                # ax.coastlines()
                # m = plt.imshow(np.flipud(obs_ok_2d), origin='upper', extent=img_extent, transform=ccrs.PlateCarree()) #, cmap=plt.cm.get_cmap('coolwarm'))
                # fig.colorbar(m)
                # plt.clim(-4, 4)
                # gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
                # gl.xlabels_top = False
                # gl.ylabels_right = False
                # ax.set_title('Kriged SST anomalies ' +str(pentad_idx)+' pentad '+str(current_year)+' year')
                # plt.show()
                # #fig.savefig('/noc/users/agfaul/ellipse_kriging/%s_%skriged.png' % (str(current_year), str(pentad_idx)))

                # Write the data.
                # This writes each time slice to the netCDF
                krig_anom[timestep, :, :] = anom.astype(
                    np.float32
                )  # ordinary_kriging
                krig_uncert[timestep, :, :] = uncert.astype(
                    np.float32
                )  # ordinary_kriging
                grid_obs[timestep, :, :] = grid_obs_2d.astype(np.float32)
                print("-- Wrote data")
                print(pentad_idx, pentad_date)

        # Write time
        # pd.date_range takes month/day/year as input dates
        clim_times_updated = [
            j.replace(year=current_year)
            for j in pd.to_datetime(clim_times.data)
        ]
        print(clim_times_updated)
        dates_ = pd.Series(clim_times_updated)
        dates = dates_.dt.to_pydatetime()  # Here it becomes date
        print("pydate", dates)

        times = date2num(dates, time.units)

        print("==== Times to be saved ====")
        print(times)

        time[:] = times
        print(time)
        # first print the Dataset object to see what we've got
        print(ncfile)
        # close the Dataset.
        ncfile.close()
        print("Dataset is closed!")


if __name__ == "__main__":
    main()
