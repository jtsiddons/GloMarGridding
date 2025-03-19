#!/usr/bin/env python
# coding: utf-8

import os

os.environ["PROJ_NETWORK"] = "OFF"

import matplotlib

matplotlib.use("Agg")

import iris
import iris.coord_categorisation as icc
import iris.analysis.maths as iam
import iris.analysis.cartography as icart
import iris.quickplot as qplt
import iris.plot as iplt
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from numpy import ma
import cartopy
import cartopy.crs as ccrs

plt.rcParams["figure.figsize"] = [12, 8]

from skimage.measure import EllipseModel
from sklearn.metrics.pairwise import euclidean_distances, haversine_distances
from sklearn.metrics import r2_score
from math import radians

# from osgeo import osr
import pandas as pd
import geopandas as geopd
from shapely.geometry import Point

import glob

from nonstationary_cov import cube_covariance as cube_cov
from nonstationary_cov import cube_io_10x10 as cube_io_10


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


def reverse_mask_from_compress_1D(
    data_cube_2D, compressed_1D_vector, fill_value=0.0, dtype=np.float32
):
    compressed_counter = 0
    print(data_cube_2D.data.mask.shape)
    err_msg = "Unexpected number of dimensions; len(shape)=" + str(
        len(data_cube_2D.data.mask.shape)
    )
    assert len(data_cube_2D.data.mask.shape) == 2, err_msg
    cube_mask = data_cube_2D.data.mask
    cube_mask_1D = cube_mask.flatten()
    ans = np.zeros_like(cube_mask_1D, dtype=dtype)
    for i in range(len(cube_mask_1D)):
        if not cube_mask_1D[i]:
            ans[i] = compressed_1D_vector[compressed_counter]
            compressed_counter += 1
    ma.set_fill_value(ans, fill_value)
    ans = ma.masked_where(cube_mask_1D, ans)
    return ans


def lat_mask(old_cube, lat_max=85.0):
    if lat_max < 90.0:
        print("Masking abs(lat) > lat_max")
        new_cube = old_cube.copy()
        _, lat_mesh = np.meshgrid(
            new_cube.coord("longitude").points,
            new_cube.coord("latitude").points,
        )
        lat_mask = np.abs(lat_mesh) >= lat_max
        new_cube.data.mask = np.logical_or(new_cube.data.mask, lat_mask)
        return new_cube
    return old_cube


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

    # Load SSTs for a reference to a sample image
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

    # Load global stiched covariance
    cor_basepath = (
        "/noc/mpoc/surface_data/ESA_CCI5deg_month_extra/ANOMALY/SpatialScales/"
    )
    cor_basepath = cor_basepath + "locally_build_covariances/"
    cor_file = (
        cor_basepath
        + "covariance_"
        + str(mm).zfill(2)
        + "_v_eq_1p5_sst_clipped.nc"
    )
    cor_cube = iris.load_cube(cor_file, "correlation")
    map_table_file = (
        cor_basepath
        + "row_lookup_"
        + str(mm).zfill(2)
        + "_v_eq_1p5_sst_without_psd_check.csv"
    )
    map_table = pd.read_csv(map_table_file)

    # Load fitted monthly ellipse file
    fitpath_base = "/noc/mpoc/surface_data/ESA_CCI5deg_month_extra/ANOMALY/"
    fitpath_base += "SpatialScales/matern_physical_distances_v_eq_1p5/"
    v = 1.5
    fitpath_dist = fitpath_base
    fitfile = fitpath_dist + "sst_" + str(mm).zfill(2) + ".nc"
    cube_matern_dist = iris.load(fitfile)
    print(cube_matern_dist)

    # Graphics options
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

    # Load stiched global covariance
    print("Reading stiched covariance/correlation")
    search_bol = (map_table.lat == latlon_point[1]) & (
        map_table.lon == latlon_point[0]
    )
    matching_rows = map_table[search_bol]
    assert_msg = "There should be only one unique matching row, nrows=" + str(
        len(matching_rows)
    )
    assert len(matching_rows) == 1, assert_msg
    matching_row = matching_rows.iloc[0]
    row_num = int(matching_row.row_num)
    correlation_vector = cor_cube.data[:, row_num]
    latlon = (matching_row.lon, matching_row.lat)
    print(latlon, correlation_vector.max(), correlation_vector.shape)

    print("Read in mask")
    # mask_template = cube_fu.extract(cube_io_10.month_2_constraint[mm])
    # R_cube = mask_template[0, :, :].copy()
    mask_template = cube_matern_dist.extract("Lx")[0]
    mask_original = mask_template.data.mask.copy()
    R_cube = mask_template[:, :].copy()
    R_cube.data.mask = False
    R_cube_x = reverse_mask_from_compress_1D(R_cube, correlation_vector)
    R_cube.data = R_cube_x.reshape(cube_shape[1:])
    R_cube.units = "1"
    R_cube.data.mask = mask_original
    print(
        np.min(R_cube.data),
        np.min(R_cube_x),
        np.max(R_cube.data),
        np.max(R_cube_x),
    )

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
    df_0p5 = pd.DataFrame({"northing": xys[:, 1], "easting": xys[:, 0]})
    points_0p5 = df_0p5.apply(
        lambda row: Point([row.easting, row.northing]), axis=1
    )
    gdf_0p5 = geopd.GeoDataFrame(df_0p5, geometry=points_0p5, crs=proj4)
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None
    ):
        print(gdf_0p5)

    # Plot global stiched correlation with projected ellipse on top
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection=proj)
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
    # title = 'Knitted SST correlation with MATA@(x='+str(latlon_point[0])+', y='+str(latlon_point[1])+') \n'
    # title += 'Month='+str(mm)+' \n'
    # title += 'with fitted ellipses, drawn to 1:'+str(scaling)+' scale'
    # ax.set_title(title, fontsize=20)
    # add_coastlines_land(ax)
    # cbar = plt.colorbar(orientation="horizontal", pad=0.02)
    # cbar.ax.tick_params(labelsize=15)
    # fig_name = fig_path+figlabel+'_'+str(mm).zfill(2)+'_stiched_corr_with_ellipse.png'
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
    print(R_cube.data.min(), R_cube.data.max())
    print(np.where(R_cube.data == R_cube.data.max()))
    cf = iplt.contourf(
        R_cube, levels=corr_contours2, cmap=corr_cmap2, extend="both"
    )
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
        "Knitted SST corr with x="
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
        + figlabel
        + "_"
        + str(mm).zfill(2)
        + "_stiched_corr_with_ellipse_correlation_contour.png"
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
