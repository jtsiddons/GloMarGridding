#!/usr/bin/env python
# coding: utf-8

import os

os.environ["PROJ_NETWORK"] = "OFF"

import matplotlib

# matplotlib.use('QtAgg')  # Requires display or xforwarding set tp work
matplotlib.use("Agg")

import iris
import iris.plot as iplt
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import numpy as np
import cartopy
import cartopy.crs as ccrs

plt.rcParams["figure.figsize"] = [12, 8]

from skimage.measure import EllipseModel

# from osgeo import osr
import pandas as pd
import geopandas as geopd
from shapely.geometry import Point

from nonstationary_cov import cube_covariance as cube_cov
from nonstationary_cov import cube_io_10x10 as cube_io_10


def guess_bounds(cube):
    for xy in ["latitude", "longitude"]:
        try:
            cube.coord(xy).guess_bounds()
        except Exception:
            pass


def add_coastlines_land(ax):
    ax.add_feature(cartopy.feature.LAND)
    ax.add_feature(cartopy.feature.COASTLINE)


def fixlon(cube):
    cube.coord("longitude").bounds = None
    lon_ge_0 = iris.Constraint(longitude=lambda val: 180.0 >= val >= 0.0)
    lon_lt_0 = iris.Constraint(longitude=lambda val: val > 180.0)
    ebuc = cube.extract(lon_ge_0)
    cbue = cube.extract(lon_lt_0)
    ebuc.coord("longitude").points = ebuc.coord("longitude").points + 360.0
    neko = iris.cube.CubeList()
    neko.append(ebuc)
    neko.append(cbue)
    neko = neko.concatenate_cube()
    neko.coord("longitude").guess_bounds()
    return neko


def mask_time_union(cube):
    cube_mask = cube.data.mask
    common_mask = np.any(cube_mask, axis=0)
    cube.data.mask = common_mask
    return cube


def draw_ellipse(latlon_point, figlabel, mm, london_in_the_middle=True):
    """Draw ellipse centre near latlon_point = (lat, lon)"""
    print(latlon_point, figlabel)

    # Config and setup variables
    varname = "sea_water_temperature_anomaly"

    # Load SSTs to compute global covariance, including a reference to a sample image
    # yyyy = 1998
    # t_index = -1
    basepath = "/noc/mpoc/surface_data/ESA_CCI5deg_month_extra/ANOMALY/"
    ncfiles = [basepath + "esa_cci_sst_5deg_monthly_1982_2022.nc"]
    print(ncfiles[0], ncfiles[-1])
    cube_fu = cube_io_10.iris_load_cube_plus(
        ncfiles, var_name=varname, fudge_negative_longitude=False
    )
    cube_fu = mask_time_union(cube_fu)
    print(repr(cube_fu))
    cube_shape = cube_fu.shape

    # Load fitted monthly ellipse file
    fitpath_base = "/noc/mpoc/surface_data/ESA_CCI5deg_month_extra/ANOMALY/"
    fitpath_base += "SpatialScales/matern_physical_distances_v_eq_1p5/"
    v = 1.5
    fitpath_dist = fitpath_base
    fitfile = fitpath_dist + "sst_" + str(mm).zfill(2) + ".nc"
    cube_matern_dist = iris.load(fitfile)
    print(cube_matern_dist)

    # Graphics options
    # sst_cmap = 'bwr'
    # sst_contours = np.linspace(-3.5, 3.5, 8)
    corr_cmap2 = mpl_cm.get_cmap("brewer_PuOr_11")
    corr_contours2 = np.array(
        [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9]
    )
    s_size = 100

    # Map projection
    if london_in_the_middle:
        central_longitude = 0.0
    else:
        central_longitude = 180.0
    proj = ccrs.PlateCarree(central_longitude=central_longitude)

    fig_path = "../test_data/"

    # Plot a sample SST frame
    # date_constraint = iris.Constraint(year=yyyy) & iris.Constraint(month_number=mm)
    # ax = plt.axes(projection=proj)
    # guess_bounds(cube_fu)
    # iplt.contourf(cube_fu.extract(date_constraint)[t_index],
    #               levels=sst_contours,
    #               extend='both',
    #               cmap=sst_cmap)
    # add_coastlines_land(plt.gca())
    # cbar = plt.colorbar(orientation="horizontal", pad=0.02)
    # cbar.ax.tick_params(labelsize=15)
    # fig_name = fig_path+'/'+figlabel+'_'+str(mm).zfill(2)+'sst_example.png'
    # plt.savefig(fig_name)
    # plt.clf()

    # Compute global covariance and correlation
    cube2cov = cube_fu.extract(cube_io_10.month_2_constraint[mm])
    print(repr(cube2cov))
    print("Creating covariance/correlation")
    cov_obj = cube_cov.CovarianceCube(cube2cov)
    print(cov_obj.Corr)
    print(cov_obj.data_has_mask)
    print(cov_obj.small_covar_size)
    xy, latlon = cov_obj.find_nearest_xy_index_in_cov_matrix(latlon_point)
    print(xy)
    print([np.round(a, 3) for a in latlon])
    print([np.round(a, 3) for a in cov_obj.xy[xy, :]])
    correlation_vector = cov_obj.Corr[xy, :]
    print(correlation_vector)

    R_cube = cube2cov[0, :, :].copy()
    R_cube_x = cov_obj._reverse_mask_from_compress_1D(cov_obj.Corr[xy, :])
    R_cube.data = R_cube_x.reshape(cube_shape[1:])
    R_cube.units = "1"

    # Get ellipse parameters
    lon_grid, lat_grid = np.meshgrid(
        R_cube.coord("longitude").points, R_cube.coord("latitude").points
    )
    latlon_pairs = np.deg2rad(
        np.column_stack([lat_grid.reshape(-1), lon_grid.reshape(-1)])
    )
    originA = np.array([latlon[1], latlon[0]])
    originA = originA[np.newaxis, :]
    origin = np.deg2rad(np.array([latlon[1], latlon[0] - 360]))
    origin = origin[np.newaxis, :]

    lon_constrain = iris.Constraint(longitude=latlon[0])
    lat_constrain = iris.Constraint(latitude=latlon[1])
    cubbie_matern_dist = cube_matern_dist.extract(lon_constrain & lat_constrain)
    print(cubbie_matern_dist)
    (Lx_dist,) = cubbie_matern_dist.extract("Lx")
    (Ly_dist,) = cubbie_matern_dist.extract("Ly")
    (theta_dist,) = cubbie_matern_dist.extract("theta")

    # Get ellise points on a local Tranverse Mercator projection
    scaling = 1
    x0, y0 = 0.0, 0.0
    params_dist = (
        x0,
        y0,
        1000.0 * float(Lx_dist.data) / scaling,
        1000.0 * float(Ly_dist.data) / scaling,
        float(theta_dist.data),
    )
    print(v, params_dist, params_dist[-1] * 180.0 / np.pi)
    xys = EllipseModel().predict_xy(
        np.linspace(-np.pi, np.pi, 360), params=params_dist
    )
    print(xys)
    gdf_lon_mean, gdf_lat_mean = str(latlon[0]), str(latlon[1])
    proj4 = "+proj=tmerc +lat_0=" + gdf_lat_mean + " +lon_0=" + gdf_lon_mean
    proj4 += " +k=0.9996012717 +x_0=0 +y_0=0 +ellps=airy +units=km +no_defs"
    proj4_cartopy = ccrs.TransverseMercator(
        central_longitude=gdf_lon_mean, central_latitude=gdf_lat_mean
    )
    df = pd.DataFrame({"northing": xys[:, 1], "easting": xys[:, 0]})
    points = df.apply(lambda row: Point([row.easting, row.northing]), axis=1)
    gdf = geopd.GeoDataFrame(df, geometry=points, crs=proj4)
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None
    ):
        print(gdf)

    # Plot global correlation with projected ellipse on top
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection=proj)
    # guess_bounds(R_cube)
    # iplt.contourf(R_cube, levels=corr_contours2, cmap=corr_cmap2, extend='both')
    # ls = []
    # linestyles = ['--']
    # v_strs = ['0.5']
    # geodfs = [gdf_0p5]
    # zipper = zip(v_strs, linestyles, geodfs)
    # for v_str, linestyle, geodf in zipper:
    #     ls += ax.plot(geodf.easting, geodf.northing,
    #                   color='blue',
    #                   linewidth=2,
    #                   linestyle=linestyle,
    #                   transform=proj4_cartopy,
    #                   label='v = '+v_str)
    # ax.scatter(latlon[0]-360, latlon[1],
    #            s=s_size,
    #            color='black',
    #            transform=ccrs.Geodetic())
    # title = 'Global SST correlation with SST@(x='+str(latlon_point[0])+', y='+str(latlon_point[1])+') \n'
    # title += 'Month='+str(mm)+' \n'
    # title += 'with fitted ellipses, drawn to 1:'+str(scaling)+' scale'
    # ax.set_title(title, fontsize=20)
    # add_coastlines_land(ax)
    # cbar = plt.colorbar(orientation="horizontal", pad=0.02)
    # cbar.ax.tick_params(labelsize=15)
    # fig_name = fig_path+'/'+figlabel+'_'+str(mm).zfill(2)+'_global_corr_with_ellipse.png'
    # plt.savefig(fig_name)
    # plt.clf()

    # Compute fitted ellipse correlation
    originAA = np.deg2rad(originA)
    distance_jj = latlon_pairs[:, 0] - originAA[0, 0]
    average_cos = 0.5 * (np.cos(latlon_pairs[:, 0]) + np.cos(originAA[0, 0]))
    distance_ii = latlon_pairs[:, 1] - originAA[0, 1]
    distance_ii[distance_ii > np.pi] = (
        distance_ii[distance_ii > np.pi] - 2.0 * np.pi
    )
    distance_ii = distance_ii * average_cos

    zonal_displacement = (
        distance_ii * cube_cov._RADIUS_OF_EARTH / cube_cov._KM2M
    )
    meridonal_displacement = (
        distance_jj * cube_cov._RADIUS_OF_EARTH / cube_cov._KM2M
    )
    meridonal_displacement = meridonal_displacement.reshape(lon_grid.shape)
    meridonal_displacement_cube = R_cube.copy()
    meridonal_displacement_cube.data = meridonal_displacement
    zonal_displacement = zonal_displacement.reshape(lon_grid.shape)
    zonal_displacement_cube = R_cube.copy()
    zonal_displacement_cube.data = zonal_displacement

    def _corr_func_dist(x_dist, y_dist):
        return cube_cov.c_ij_anistropic_rotated(
            v,
            1.0,
            x_dist,
            y_dist,
            float(Lx_dist.data),
            float(Ly_dist.data),
            float(theta_dist.data),
        )

    corr_func_dist = np.vectorize(_corr_func_dist)
    predicted_corr_dist = corr_func_dist(
        zonal_displacement_cube.data, meridonal_displacement_cube.data
    )
    predicted_corr_dist_cube = R_cube.copy()
    predicted_corr_dist_cube.data = predicted_corr_dist
    predicted_corr_dist_cube2 = predicted_corr_dist_cube.copy()
    predicted_corr_dist_cube2.data = np.ma.masked_where(
        R_cube.data.mask, predicted_corr_dist_cube2.data
    )
    # Overlay fitted ellipse correlation with observed
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    guess_bounds(R_cube)
    cf = iplt.contourf(
        R_cube, levels=corr_contours2, cmap=corr_cmap2, extend="both"
    )
    guess_bounds(predicted_corr_dist_cube2)
    cl = iplt.contour(
        predicted_corr_dist_cube2, levels=corr_contours2, colors="k"
    )
    ax.scatter(
        latlon[0] - 360,
        latlon[1],
        s=s_size,
        color="black",
        transform=ccrs.Geodetic(),
    )
    title = (
        "SST corr with x="
        + str(latlon_point[0])
        + " y="
        + str(latlon_point[1])
        + " \n"
    )
    title += "Month=" + str(mm) + " with ellipse correlation in black contours"
    ax.set_title(title, fontsize=20)
    add_coastlines_land(ax)
    cbar = plt.colorbar(cf, orientation="horizontal", pad=0.02)
    cbar.ax.tick_params(labelsize=15)
    plt.clabel(cl, fontsize=15, inline=True)
    fig_name = (
        fig_path
        + "/"
        + figlabel
        + "_"
        + str(mm).zfill(2)
        + "_global_corr_with_ellipse_correlation_contour.png"
    )
    plt.savefig(fig_name)
    plt.clf()

    # Finish!
    print(figlabel + " completed.")


def main():
    mms = [1, 7]
    lonlat_points = [(-172.5, 2.5), (-62.5, 12.5), (-12.5, 32.5)]
    figlabels = ["Central_Eq_Pacific", "Carribbean", "Canary_Islands"]
    london_in_the_middles = [False, True, True]
    for mm in mms:
        for lonlat_pt, figlabel, london_in_the_middle in zip(
            lonlat_points, figlabels, london_in_the_middles
        ):
            draw_ellipse(lonlat_pt, figlabel, mm, london_in_the_middle)


if __name__ == "__main__":
    main()
