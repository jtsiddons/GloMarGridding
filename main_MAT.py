#!/usr/bin/env python  # noqa: D100

################
# by A. Faulkner
# for python version 3.0 and up
################

# argument parser
import argparse
import os
from datetime import date
import netCDF4 as nc
import numpy as np
import polars as pl
import xarray as xr
import yaml
from netCDF4 import date2num

# PyCOADS functions
from PyCOADS.processing.solar import is_daytime

import glomar_gridding.error_covariance as err_cov
import glomar_gridding.kriging as krig
import glomar_gridding.observations as obs_module
from glomar_gridding.climatology import join_climatology_by_doy
from glomar_gridding.grid import assign_to_grid
from glomar_gridding.interpolation_covariance import load_covariance
from glomar_gridding.io import load_array, load_dataset
from glomar_gridding.mask import mask_observations
from glomar_gridding.matern_and_tm.matern_tau import tau_dist
from glomar_gridding.utils import get_pentad_range

# NOC Specific Helper Fucntions
from .noc_helpers import (
    add_height_adjustment,
    merge_ellipse_params,
    load_icoads,
)

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
    help="height member: if height member is 0, no height adjustment is added.",
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


def _measurement_covariance(
    df: pl.DataFrame,
    sig_ms: float,
    sig_mb: float,
    sig_bs: float,
    sig_bb: float,
) -> tuple[np.ndarray, np.ndarray]:
    obs_bias_map = {"ship": sig_ms, "buoy": sig_mb}
    cov = err_cov.uncorrelated_components(
        df, group_col="data_type", obs_sig_map=obs_bias_map
    )
    dist, weights = err_cov.dist_weight(df, dist_fn=tau_dist)
    cov = cov + dist
    del dist
    # print(covx1, covx1.shape)
    bias_uncert_map = {"ship": sig_bs, "buoy": sig_bb}
    cov = cov + err_cov.correlated_components(
        df,
        group_col="data_type",
        bias_sig_map=bias_uncert_map,
    )
    return cov, weights


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
    qc_dir = config.get("observations", {}).get("icoads_qc_flags")

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
    # Do we need this if we are computing later
    pentad_climatology = obs_module.read_climatology(
        metoffice_climatology, lat_south, lat_north, lon_west, lon_east
    )
    clim_times = (
        pl.from_numpy(pentad_climatology.time).to_series().cast(pl.Datetime)
    )
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
        del by_month_frame, yr_rng

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

            # interpolation covariance
            covariance: np.ndarray = load_covariance(
                cov_dir, month=current_month
            )
            print(covariance)
            # INFO: Adjust diagonal to ensure positive definiteness
            diag_ind = np.diag_indices_from(covariance)
            covariance[diag_ind] = covariance[diag_ind] * 1.01 + 0.005
            print(covariance)

            mask_ds: xr.DataArray = load_array(
                mask_dir, var="mask", month=current_month
            )
            print(mask_ds)

            # read in observations and QC
            obs_df = load_icoads(
                data_dir,
                qc_dir,
                var="at",
                year=current_year,
                month=current_month,
            )
            obs_df = obs_df.pipe(is_daytime).filter(pl.col("is_daytime").eq(0))
            print(obs_df)  # [['local_datetime', 'is_daytime']])
            # read in climatology here
            # match with obs against DOY
            print(clim)
            # obs_df = obs_qc_module.MAT_match_climatology(obs_df, clim)
            obs_df: pl.DataFrame = join_climatology_by_doy(obs_df, clim)

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

            # read in ellipse parameters file corresponding to the processed
            # file
            month_ellipse_param = load_dataset(
                ellipse_param_dir, month=current_month
            )

            # list of dates for each year
            # _, month_range = monthrange(current_year, current_month)
            # print(month_range)

            # if we do MetOffice processing:
            for timestep, current_date in enumerate(monthly):
                print(f"Doing iteration {timestep} for date {current_date}")

                start_date, end_date = get_pentad_range(current_date)
                pentad_df = obs_df.filter(
                    pl.col("datetime").is_between(
                        start_date, end_date, closed="both"
                    )
                )

                # calculate flattened idx based on the ESA landmask file
                # Align the observations to the mask
                pentad_df = mask_observations(
                    pentad_df,
                    mask_ds,
                    varnames="at",
                    align_to_mask=True,
                )

                # plotting_df = cond_df[
                #     ["lon", "lat", "sst", "climatology_sst", "sst_anomaly"]
                # ]
                # lons = plotting_df["lon"]
                # lats = plotting_df["lat"]
                # ssts = plotting_df["sst"]
                # clims = plotting_df["climatology_sst"]
                # anoms = plotting_df["sst_anomaly"]
                #
                # skwargs = {"s": 2, "c": "red"}
                # fig = plt.figure(figsize=(10, 5))
                # ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
                # cp.projected_scatter(
                #     fig,
                #     ax,
                #     lons,
                #     lats,
                #     skwargs=skwargs,
                #     title="ICOADS locations - "
                #     + str(pentad_idx)
                #     + " pentad "
                #     + str(current_year)
                #     + " year",
                # )
                # # plt.show()
                # fig.savefig(
                #     "/noc/users/agfaul/ellipse_kriging/%s_%spoints.png"
                #     % (str(current_year), str(pentad_idx))
                # )
                #
                # skwargs = {
                #     "s": 2,
                #     "c": ssts,
                #     "cmap": plt.cm.get_cmap("coolwarm"),
                #     "clim": (-10, 14),
                # }
                # ckwargs = {"label": "SST [deg C]"}
                # fig = plt.figure(figsize=(10, 5))
                # ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
                # cp.projected_scatter(
                #     fig,
                #     ax,
                #     lons,
                #     lats,
                #     add_colorbar=True,
                #     skwargs=skwargs,
                #     ckwargs=ckwargs,
                #     title="ICOADS measured SST -"
                #     + str(pentad_idx)
                #     + " pentad "
                #     + str(current_year)
                #     + " year",
                #     land_col="darkolivegreen",
                # )
                # # plt.show()
                # fig.savefig(
                #     "/noc/users/agfaul/ellipse_kriging/%s_%ssst.png"
                #     % (str(current_year), str(pentad_idx))
                # )
                #
                # skwargs = {
                #     "s": 2,
                #     "c": clims,
                #     "cmap": plt.cm.get_cmap("coolwarm"),
                #     "clim": (-10, 14),
                # }
                # ckwargs = {"label": "SST [deg C]"}
                # fig = plt.figure(figsize=(10, 5))
                # ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
                # cp.projected_scatter(
                #     fig,
                #     ax,
                #     lons,
                #     lats,
                #     add_colorbar=True,
                #     skwargs=skwargs,
                #     ckwargs=ckwargs,
                #     title="ESA CCI climatology - "
                #     + str(pentad_idx)
                #     + " pentad "
                #     + str(current_year)
                #     + " year",
                #     land_col="darkolivegreen",
                # )
                # # plt.show()
                # fig.savefig(
                #     "/noc/users/agfaul/ellipse_kriging/%s_%sclim.png"
                #     % (str(current_year), str(pentad_idx))
                # )
                #
                # skwargs = {
                #     "s": 2,
                #     "c": anoms,
                #     "cmap": plt.cm.get_cmap("coolwarm"),
                #     "clim": (-10, 14),
                # }
                # ckwargs = {"label": "SST [deg C]"}
                # fig = plt.figure(figsize=(10, 5))
                # ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
                # cp.projected_scatter(
                #     fig,
                #     ax,
                #     lons,
                #     lats,
                #     add_colorbar=True,
                #     skwargs=skwargs,
                #     ckwargs=ckwargs,
                #     title="ICOADS SST anomalies - "
                #     + str(pentad_idx)
                #     + " pentad "
                #     + str(current_year)
                #     + " year",
                #     land_col="darkolivegreen",
                # )
                # # plt.show()
                # fig.savefig(
                #     "/noc/users/agfaul/ellipse_kriging/%s_%sanom.png"
                #     % (str(current_year), str(pentad_idx))
                # )
                #
                # day_flat_idx = day_df.get_column(["flattened_idx"])

                # match gridded observations to ellipse parameters
                pentad_df = merge_ellipse_params(month_ellipse_param, pentad_df)

                gridbox_counts = pentad_df["grid_idx"].value_counts()
                grid_obs_2d: np.ndarray = assign_to_grid(
                    gridbox_counts["count"].to_numpy(),
                    gridbox_counts["grid_idx"].to_numpy(),
                    mask_ds["landice_sea_mask"],
                ).values
                del gridbox_counts

                # Error Covariance
                obs_covariance, W = _measurement_covariance(
                    pentad_df, sig_ms, sig_mb, sig_bs, sig_bb
                )

                # krige obs onto gridded field
                grid_idx_with_obs = (
                    pentad_df["grid_idx"].unique().sort().to_numpy()
                )

                anom, uncert = krig.kriging(
                    obs_idx=grid_idx_with_obs,
                    weights=W,
                    obs=pentad_df["anomaly"].to_numpy(),
                    interp_cov=covariance,
                    error_cov=obs_covariance,
                    method=method,
                )
                anom = assign_to_grid(
                    anom, grid_idx_with_obs, mask_ds["landice_sea_mask"]
                ).values
                uncert = assign_to_grid(
                    uncert, grid_idx_with_obs, mask_ds["landice_sea_mask"]
                ).values
                print("Kriging done, saving output")

                # fig = plt.figure(figsize=(10, 5))
                # img_extent = (-180.0, 180.0, -90.0, 90.0)
                # ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
                # ax.set_extent(
                #     [-180.0, 180.0, -90.0, 90.0], crs=ccrs.PlateCarree()
                # )
                # ax.add_feature(cfeature.LAND, color="darkolivegreen")
                # ax.coastlines()
                # m = plt.imshow(
                #     np.flipud(obs_ok_2d),
                #     origin="upper",
                #     extent=img_extent,
                #     transform=ccrs.PlateCarree(),
                # )  # , cmap=plt.cm.get_cmap('coolwarm'))
                # fig.colorbar(m)
                # plt.clim(-4, 4)
                # gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
                # gl.xlabels_top = False
                # gl.ylabels_right = False
                # ax.set_title(
                #     "Kriged SST anomalies "
                #     + str(pentad_idx)
                #     + " pentad "
                #     + str(current_year)
                #     + " year"
                # )
                # plt.show()
                # fig.savefig(
                #     "/noc/users/agfaul/ellipse_kriging/%s_%skriged.png"
                #     % (str(current_year), str(pentad_idx))
                # )

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
                print(timestep, current_date)

        # Write time
        # pd.date_range takes month/day/year as input dates
        dates = clim_times.dt.replace(year=current_year)

        time[:] = date2num(dates, time.units)
        print(time)
        # first print the Dataset object to see what we've got
        print(ncfile)
        # close the Dataset.
        ncfile.close()
        print("Dataset is closed!")


if __name__ == "__main__":
    main()
