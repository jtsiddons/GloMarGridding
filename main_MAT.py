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
from calendar import isleap, monthrange

# import datetime as dt
from datetime import timedelta
from netCDF4 import date2num

# data handling tools
import pandas as pd
import netCDF4 as nc
import polars as pl

# self-written modules (from the same directory)
import glomar_gridding.covariance as cov_module
import glomar_gridding.observations as obs_module
import glomar_gridding.observations_plus_qc as obs_qc_module
import glomar_gridding.kriging as krig_module

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


def main():  # noqa: D103
    config, year_start, year_stop, height_member, method = _parse_args(parser)

    # load config options from ini file
    # this is done using an ini config file, which is located in the same direcotry as the python code
    # instantiate

    print(config)

    # read values from auxiliary_files section
    # for string use config.get
    # for boolean use config.getboolean
    # for int use config.getint
    # for float use config.getfloat
    # for list (multiple options for same key) use config.getlist

    # location of MAT climatology
    climatology = config.get("climatology", {}).get("path")
    metoffice_climatology = config.get("observations", "metoffice_climatology")

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
    data_path: str = config.get("observations", {}).get("path")

    height_adjustment_path: str | None = None
    adjusted_height: int | None = None
    if height_member > 0:
        height_adjustment_path = config.get("parameters", {}).get(
            "height_adjustments"
        )
        adjusted_height = config.get("parameters", {}).get("adjusted_height")

    # location og QC flags in GROUPS subdirectories
    qc_mat = config.get("observations", {}).get("qc")
    qc_path = config.get("observations", {}).get("icoads_qc_flags")
    qc_path_2 = config.get("observations", {}).get("icoads_qc_flags_tracked")

    # path where the covariance(s) is/are located
    # if single covariance, then full path
    # if several different covariances, then path to directory
    cov_dir = config.get("covariance", {}).get("path")

    output_directory = config.get("covariance", {}).get("output")

    ellipse_param_path = config.get("covariance", {}).get("ellipse_parameters")

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
    # while doing pentad processing, this will set "mid-pentads" dates for the year
    pentad_climatology = obs_module.read_climatology(
        metoffice_climatology, lat_south, lat_north, lon_west, lon_east
    )
    clim_times = pentad_climatology.time
    print(clim_times)
    del pentad_climatology
    # climatology2 = np.broadcast_to(mask_ds.landmask.values > 0, climatology.climatology.values.shape)

    year_list = range(year_start, year_stop + 1)
    month_list = range(1, 13, 1)

    for current_year in year_list:
        # add MetOffice pentads here
        yr_rng = pd.date_range("1970/01/03", "1970/12/31", freq="5D")

        times2 = [j.replace(year=current_year) for j in yr_rng]
        print(times2)
        times_series = pd.Series(times2)
        by_month = list(
            times_series.groupby(times_series.map(lambda x: x.month))
        )
        print(by_month)

        try:
            ncfile.close()  # make sure dataset is not already open.
        except (NameError, RuntimeError):
            # ncfile not created yet, or already closed
            pass
        except Exception as e:  # Unknown Error
            raise e

        ncfilename = str(output_directory)
        ncfilename = f"{current_year}_{method}_kriged_MAT"
        if height_member:
            # QUESTION: why not the adjusted height value in the filename?
            ncfilename += f"_heightmember_{height_member:03d}m"
        # if adjusted_height:
        #     ncfilename += f"_height_{adjusted_height:.2f}m"
        ncfilename += ".nc"
        ncfilename = os.path.join(output_directory, ncfilename)

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

            idx, monthly = by_month[current_month - 1]

            print(monthly)
            print(idx)
            print(monthly.index)
            print("Current month and year: ", (current_month, current_year))

            # covariance = cov_module.read_in_covarance_file(cov_dir, month=current_month)
            covariance = cov_module.get_covariance(cov_dir, month=current_month)
            print(covariance)
            diag_ind = np.diag_indices_from(covariance)
            covariance[diag_ind] = covariance[diag_ind] * 1.01 + 0.005
            print(covariance)

            # WARN: Should this be a landmask file rather than covariance?
            mask_ds, mask_ds_lat, mask_ds_lon = cov_module.get_landmask(
                cov_dir, month=current_month
            )
            print(mask_ds)

            # read in observations and QC
            obs_df = obs_qc_module.MAT_main(
                data_path,
                qc_path,
                qc_path_2,
                qc_mat,
                year=current_year,
                month=current_month,
            )
            day_night = pl.from_pandas(
                obs_df.loc["uid", "datetime", "lon", "lat"]
            )  # required cols for is_daytime
            day_night = day_night.pipe(is_daytime)
            obs_df = obs_df.merge(
                day_night.select(["uid", "is_daytime"]).to_pandas(), on="uid"
            )
            del day_night

            # filter day (1) or night(0)
            obs_df = obs_df[obs_df["is_daytime"] == 0]
            print(obs_df)  # [['local_datetime', 'is_daytime']])
            # read in climatology here
            # match with obs against DOY
            print(clim)
            # obs_df = obs_qc_module.MAT_match_climatology(obs_df, clim)
            obs_df = obs_qc_module.MAT_match_climatology_to_obs(clim, obs_df)

            # merge on the height adjustment
            if (
                height_member > 0
                and adjusted_height is not None
                and height_adjustment_path is not None
            ):
                obs_df = obs_qc_module.MAT_add_height_adjustment(
                    obs_df,
                    height_adjustment_path=height_adjustment_path,
                    year=current_year,
                    adjusted_height=adjusted_height,
                    height_member=height_member,
                    mat_col="obs_anomalies",
                )

            print(obs_df)
            print(obs_df.columns.values)

            # read in ellipse parameters file corresponding to the processed file
            month_ellipse_param = obs_qc_module.ellipse_param(
                ellipse_param_path, month=current_month, var="MAT"
            )

            # list of dates for each year
            _, month_range = monthrange(current_year, current_month)
            # print(month_range)

            # if we do MetOffice processing:
            for pentad_idx, pentad_date in enumerate(monthly):
                print(pentad_date)
                print(pentad_idx)

                timestep = pentad_idx
                current_date = pentad_date

                if isleap(current_year):
                    fake_non_leap_year = 1970
                    current_date = current_date.replace(year=fake_non_leap_year)
                    start_date = current_date - timedelta(days=2)
                    end_date = current_date + timedelta(days=2)
                    start_date = start_date.replace(year=current_year)
                    end_date = end_date.replace(year=current_year)
                    print("----------")
                    print("timestep", timestep)
                    print("current date", current_date)
                    print("start date", start_date)
                    print("end date", end_date)
                    print("----------")
                    day_df = obs_df.loc[
                        (obs_df["datetime"] >= str(start_date))
                        & (obs_df["datetime"] <= str(end_date))
                    ]
                else:
                    start_date = current_date - timedelta(days=2)
                    end_date = current_date + timedelta(days=2)
                    print("----------")
                    print("timestep", timestep)
                    print("current date", current_date)
                    print("start date", start_date)
                    print("end date", end_date)
                    print("----------")
                    day_df = obs_df.loc[
                        (obs_df["datetime"] >= str(start_date))
                        & (obs_df["datetime"] <= str(end_date))
                    ]

                print(f"{day_df =}")

                # calculate flattened idx based on the ESA landmask file
                # which is compatible with the ESA-derived covariance
                # mask_ds, mask_ds_lat, mask_ds_lon = obs_module.landmask(water_mask_file, lat_south,lat_north, lon_west,lon_east)
                cond_df, obs_flat_idx = obs_module.watermask_at_obs_locations(
                    lon_bnds,
                    lat_bnds,
                    day_df,
                    mask_ds,
                    mask_ds_lat,
                    mask_ds_lon,
                )
                cond_df.reset_index(drop=True, inplace=True)

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

                day_flat_idx = cond_df["flattened_idx"][:]

                # match gridded observations to ellipse parameters
                cond_df = obs_module.match_ellipse_parameters_to_gridded_obs(
                    month_ellipse_param, cond_df, mask_ds
                )

                cond_df["gridbox"] = day_flat_idx  # .values.reshape(-1)
                gridbox_counts = cond_df["gridbox"].value_counts()
                gridbox_count_np = gridbox_counts.to_numpy()
                gridbox_id_np = gridbox_counts.index.to_numpy()
                del gridbox_counts
                water_mask = np.copy(
                    mask_ds.variables["landice_sea_mask"][:, :]
                )
                grid_obs_2d = krig_module.result_reshape_2d(
                    gridbox_count_np, gridbox_id_np, water_mask
                )

                obs_covariance, W = obs_module.measurement_covariance(
                    cond_df, day_flat_idx, sig_ms, sig_mb, sig_bs, sig_bb
                )
                print(obs_covariance)
                print(W)

                # krige obs onto gridded field
                anom, uncert = krig_module.kriging_main(
                    covariance,
                    cond_df,
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
