#!/usr/bin/env python

################
# by A. Faulkner
# for python version 3.0 and up
################

# global
import os

# argument parser
import argparse
from configparser import ConfigParser

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
from glomar_gridding.utils import ConfigParserMultiValues

# PyCOADS functions
from PyCOADS.processing.solar import is_daytime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-config",
        dest="config",
        required=False,
        default="config.ini",
        help="INI file containing configuration settings",
    )
    parser.add_argument(
        "-year_start", dest="year_start", required=False, help="start year"
    )
    parser.add_argument(
        "-year_stop", dest="year_stop", required=False, help="end year"
    )
    parser.add_argument(
        "-month", dest="month", required=True, help="month"
    )  # New Argument
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
    args = parser.parse_args()

    config_file = args.config
    print(config_file)

    # load config options from ini file
    # this is done using an ini config file, which is located in the same direcotry as the python code
    # instantiate
    # ===== MODIFIED =====
    config = ConfigParser(
        strict=False,
        empty_lines_in_values=False,
        dict_type=ConfigParserMultiValues,
        converters={"list": ConfigParserMultiValues.getlist},
    )
    # ===== MODIFIED =====
    # parce existing config file
    config.read(config_file)  # ('config.ini' or 'three_step_kriging.ini')

    print(config)

    # read values from auxiliary_files section
    # for string use config.get
    # for boolean use config.getboolean
    # for int use config.getint
    # for float use config.getfloat
    # for list (multiple options for same key) use config.getlist

    # location of MAT climatology
    climatology = config.get("MAT", "climatology")
    metoffice_climatology = config.get("observations", "metoffice_climatology")

    # set boundaries for the domain
    lon_west = config.getfloat("MAT", "lon_west")  # -180.
    lon_east = config.getfloat("MAT", "lon_east")  # 180.
    lat_south = config.getfloat("MAT", "lat_south")  # -90.
    lat_north = config.getfloat("MAT", "lat_north")  # 90.

    # location of the ICOADS observation files
    data_path = config.get("MAT", "observations")

    member = args.height_member
    print(member)
    if member > 0:
        height_adjustment_path = config.get("MAT", "height_adjustments")
        adjusted_height = config.getint("MAT", "adjusted_height")

    # location og QC flags in GROUPS subdirectories
    qc_mat = config.get("MAT", "qc")
    qc_path = config.get("observations", "qc_flags_joe")
    qc_path_2 = config.get("observations", "qc_flags_joe_tracked")

    if args.year_start and args.year_stop:
        year_start = int(args.year_start)
        year_stop = int(args.year_stop)
    else:
        # start_date
        year_start = config.getint("MAT", "startyear")
        # end_date
        year_stop = config.getint("MAT", "endyear")
    print(year_start, year_stop)

    # ===== NEW =====
    mm2process = int(args.month)
    # ===== NEW =====

    # path where the covariance(s) is/are located
    # if single covariance, then full path
    # if several different covariances, then path to directory
    cov_dir = config.get("MAT", "covariance")

    output_directory = config.get("MAT", "output")

    ellipse_param_path = config.get("MAT", "ellipse_parameters")

    bnds = [lon_west, lon_east, lat_south, lat_north]
    # extract the latitude and longitude boundaries from user input
    lon_bnds, lat_bnds = (bnds[0], bnds[1]), (bnds[2], bnds[3])
    print(lon_bnds, lat_bnds)

    output_lat = np.arange(lat_bnds[0] + 0.5, lat_bnds[-1] + 0.5, 1)
    output_lon = np.arange(lon_bnds[0] + 0.5, lon_bnds[-1] + 0.5, 1)
    print(output_lat)
    print(output_lon)

    """
    #land-water-mask for observations
    water_mask_dir = config.get('MAT', 'covariance')
    mask_ds, mask_ds_lat, mask_ds_lon = cov_module.get_landmask(water_mask_dir, month=1)
    """

    """
    water_mask_file = config.getlist('landmask', 'land_mask')
    print(water_mask_file)
    #for ellipse Atlatic 0, for ellipse world 1, for ESA world 2
    water_mask_file = water_mask_file[1]
    print(water_mask_file)
    
    mask_ds, mask_ds_lat, mask_ds_lon = obs_module.landmask(water_mask_file, lat_south,lat_north, lon_west,lon_east)
    print('----')
    print(mask_ds)
    """

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

    year_list = list(range(int(year_start), int(year_stop) + 1, 1))
    # ===== MODIFIED =====
    month_list = list(range(mm2process, mm2process + 1, 1))
    # ===== MODIFIED =====

    for i in range(len(year_list)):
        current_year = year_list[i]

        # add MetOffice pentads here
        yr_rng = pd.date_range("1970/01/03", "1970/12/31", freq="5D")

        # ===== NEW =====
        yr_rng = yr_rng[yr_rng.month == mm2process]
        # ===== NEW =====

        times2 = [j.replace(year=current_year) for j in yr_rng]
        print(times2)
        times_series = pd.Series(times2)
        by_month = list(
            times_series.groupby(times_series.map(lambda x: x.month))
        )
        print(by_month)

        try:
            ncfile.close()  # make sure dataset is not already open.
        except:
            pass

        ncfilename = str(output_directory)
        ncfilename = f"{current_year}_{mm2process:02d}_kriged_MAT"
        if member:
            ncfilename += f"_heightmember_{member:03d}"
        ncfilename += ".nc"
        ncfilename = os.path.join(output_directory, ncfilename)

        ncfile = nc.Dataset(ncfilename, mode="w", format="NETCDF4_CLASSIC")
        # print(ncfile)

        lat_dim = ncfile.createDimension(
            "lat", len(output_lat)
        )  # latitude axis
        lon_dim = ncfile.createDimension(
            "lon", len(output_lon)
        )  # longitude axis
        time_dim = ncfile.createDimension("time", None)  # unlimited axis

        # Define two variables with the same names as dimensions,
        # a conventional way to define "coordinate variables".
        # ===== MODIFIED =====
        lat = ncfile.createVariable("lat", np.float32, ("lat",))
        lat.units = "degrees_north"
        lat.long_name = "latitude"
        lon = ncfile.createVariable("lon", np.float32, ("lon",))
        lon.units = "degrees_east"
        lon.long_name = "longitude"
        # ===== MODIFIED =====
        time = ncfile.createVariable("time", np.float32, ("time",))
        time.units = "days since %s-01-01" % (str(current_year))
        time.long_name = "time"
        # print(time)

        # Define a 3D variable to hold the data
        if member:
            krig_anom = ncfile.createVariable(
                f"mat_anomaly_{adjusted_height}m",
                np.float32,
                ("time", "lat", "lon"),
            )
            # note: unlimited dimension is leftmost
            krig_anom.standard_name = f"MAT anomaly at {adjusted_height} m"
            krig_anom.height = str(adjusted_height)
            krig_anom.ensemble_member = member
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

        # Write latitudes, longitudes.
        # Note: the ":" is necessary in these "write" statements
        lat[:] = output_lat  # ds.lat.values
        lon[:] = output_lon  # ds.lon.values

        for j in range(len(month_list)):
            current_month = month_list[j]
            # print(current_month)

            # ===== MODIFIED =====
            idx, monthly = by_month[0]
            # ===== MODIFIED =====

            print(monthly)
            print(idx)
            print(monthly.index)

            # ===== NEW =====
            # Print some helpful info, including current date and memory usage
            print("Current month and year: ", (current_month, current_year))

            # gs = dir()
            # sg = globals()
            # mem_update = [(x, sys.getsizeof(sg.get(x))/1E9) for x in gs if not x.startswith('_') and x not in sys.modules and x not in gs]
            # mem_update = sorted(mem_update, key=lambda x: x[1], reverse=True)
            # print('Memory usage update:')
            # print(mem_update)
            # for asdf, etadpu_mem in enumerate(mem_update):
            #     print(asdf, etadpu_mem)
            #     if asdf >= 19:
            #         break
            # ===== NEW =====

            # covariance = cov_module.read_in_covarance_file(cov_dir, month=current_month)
            covariance = cov_module.get_covariance(cov_dir, month=current_month)
            print(covariance)
            diag_ind = np.diag_indices_from(covariance)
            covariance[diag_ind] = covariance[diag_ind] * 1.01 + 0.005
            print(covariance)

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
            print(obs_df)
            day_night = pl.from_pandas(
                obs_df[["uid", "datetime", "lon", "lat"]]
            )  # required cols for is_daytime
            day_night = day_night.pipe(is_daytime)
            obs_df = obs_df.merge(
                day_night.select(["uid", "is_daytime"]).to_pandas(), on="uid"
            )
            del day_night
            """
            #use day/night mask from PyCOADS
            obs_df_polars = pl.from_pandas(obs_df)  # obs_df is a pandas frame
            daynight_obs_df_polars = is_daytime(obs_df_polars)
            obs_df = daynight_obs_df_polars.to_pandas()
            print(obs_df)
            """
            # filter day (1) or night(0)
            obs_df = obs_df[obs_df["is_daytime"] == 0]
            print(obs_df)  # [['local_datetime', 'is_daytime']])
            # read in climatology here
            # match with obs against DOY
            print(clim)
            # obs_df = obs_qc_module.MAT_match_climatology(obs_df, clim)
            obs_df = obs_qc_module.MAT_match_climatology_to_obs(clim, obs_df)

            # merge on the height adjustment
            if member > 0:
                obs_df = obs_qc_module.MAT_add_height_adjustment(
                    obs_df,
                    height_adjustment_path=height_adjustment_path,
                    year=current_year,
                    adjusted_height=adjusted_height,
                    height_member=member,
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
            for i in monthly.index:
                print("i", i, "monthly.index", monthly.index)
                pentad_date = monthly[i]
                pentad_idx = i
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

                print(day_df)

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
                # print(cond_df.columns.values)
                # print(cond_df[['lat', 'lon', 'flattened_idx', 'sst', 'climatology_sst', 'sst_anomaly']])
                # quick temperature check
                # print(cond_df['sst'])
                # print(cond_df['climatology_sst'])
                # print(cond_df['sst_anomaly'])

                """
                plotting_df = cond_df[['lon', 'lat', 'sst', 'climatology_sst', 'sst_anomaly']]
                lons = plotting_df['lon']
                lats = plotting_df['lat']
                ssts = plotting_df['sst']
                clims = plotting_df['climatology_sst']
                anoms = plotting_df['sst_anomaly']
                
                skwargs = {'s': 2, 'c': 'red'}
                fig = plt.figure(figsize=(10, 5))
                ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
                cp.projected_scatter(fig, ax, lons, lats, skwargs=skwargs, title='ICOADS locations - '+str(pentad_idx)+' pentad '+ str(current_year)+' year')
                #plt.show()
                fig.savefig('/noc/users/agfaul/ellipse_kriging/%s_%spoints.png' % (str(current_year), str(pentad_idx)))
                
                skwargs = {'s': 2, 'c': ssts, 'cmap': plt.cm.get_cmap('coolwarm'), 'clim': (-10, 14)}
                ckwargs = {'label': 'SST [deg C]'}
                fig = plt.figure(figsize=(10, 5))
                ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
                cp.projected_scatter(fig, ax, lons, lats, add_colorbar=True, skwargs=skwargs, ckwargs=ckwargs, title='ICOADS measured SST -' +str(pentad_idx)+ ' pentad ' + str(current_year)+' year', land_col='darkolivegreen')
                #plt.show()
                fig.savefig('/noc/users/agfaul/ellipse_kriging/%s_%ssst.png' % (str(current_year), str(pentad_idx)))
                
                skwargs = {'s': 2, 'c': clims, 'cmap': plt.cm.get_cmap('coolwarm'), 'clim': (-10, 14)}
                ckwargs = {'label': 'SST [deg C]'}
                fig = plt.figure(figsize=(10, 5))
                ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
                cp.projected_scatter(fig, ax, lons, lats, add_colorbar=True, skwargs=skwargs, ckwargs=ckwargs, title='ESA CCI climatology - '+str(pentad_idx)+' pentad '+ str(current_year)+' year', land_col='darkolivegreen')
                #plt.show()
                fig.savefig('/noc/users/agfaul/ellipse_kriging/%s_%sclim.png' % (str(current_year), str(pentad_idx)))
                    
                skwargs = {'s': 2, 'c': anoms, 'cmap': plt.cm.get_cmap('coolwarm'), 'clim': (-10, 14)}
                ckwargs = {'label': 'SST [deg C]'}
                fig = plt.figure(figsize=(10, 5))
                ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
                cp.projected_scatter(fig, ax, lons, lats, add_colorbar=True, skwargs=skwargs, ckwargs=ckwargs, title='ICOADS SST anomalies - '+str(pentad_idx)+ ' pentad '+ str(current_year)+' year', land_col='darkolivegreen')
                #plt.show()
                fig.savefig('/noc/users/agfaul/ellipse_kriging/%s_%sanom.png' % (str(current_year), str(pentad_idx)))
                """

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
                    cond_df,
                    day_flat_idx,
                    sig_ms=0.73,
                    sig_mb=0.24,
                    sig_bs=1.47,
                    sig_bb=0.38,
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
                    kriging_method=args.method,
                )
                print("Kriging done, saving output")

                """
                fig = plt.figure(figsize=(10, 5))
                img_extent = (-180., 180., -90., 90.)
                ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
                ax.set_extent([-180., 180., -90., 90.], crs=ccrs.PlateCarree())
                ax.add_feature(cfeature.LAND, color='darkolivegreen')
                ax.coastlines()
                m = plt.imshow(np.flipud(obs_ok_2d), origin='upper', extent=img_extent, transform=ccrs.PlateCarree(), cmap=plt.cm.get_cmap('coolwarm'))
                fig.colorbar(m)
                plt.clim(-4, 4)
                gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
                gl.xlabels_top = False
                gl.ylabels_right = False
                ax.set_title('Kriged SST anomalies ' +str(pentad_idx)+' pentad '+str(current_year)+' year')
                plt.show()
                #fig.savefig('/noc/users/agfaul/ellipse_kriging/%s_%skriged.png' % (str(current_year), str(pentad_idx)))
                """
                # Write the data.
                # This writes each time slice to the netCDF
                krig_anom[timestep, :, :] = anom.astype(
                    np.float32
                )  # ordinary_kriging
                krig_uncert[timestep, :, :] = uncert.astype(
                    np.float32
                )  # ordinary_kriging
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

        # ===== NEW + MODIFIED =====
        new_dates = []
        for dddd in dates:
            if dddd.month == mm2process:
                new_dates.append(dddd)
        print("pydate2", new_dates)

        times = date2num(new_dates, time.units)
        # ===== NEW + MODIFIED =====

        print("==== Times to be saved ====")
        print(times)

        """
        #dates_ = pd.date_range(str(current_month)+'/3/'+str(current_year), str(current_month)+'/28/'+str(current_year), freq='5D')
        #dates_ = pd.Series([datetime.combine(i, datetime.min.time()) for i in dates_])
        print('dates', dates_)
        dates = dates_.dt.to_pydatetime() # Here it becomes date
        print('pydate', dates)
        times = date2num(dates, time.units)
        print(times)
        """
        time[:] = times
        print(time)
        # first print the Dataset object to see what we've got
        print(ncfile)
        # close the Dataset.
        ncfile.close()
        print("Dataset is closed!")


if __name__ == "__main__":
    main()
