import pandas as pd
import geopandas as gpd
import numpy as np
from scipy import stats as sstats
from scipy.spatial.transform import Rotation
from shapely.geometry import Point


def latlon2ne(latlons,
              latlons_in_rads=False,
              latlon0=(0.0, 180.0)):
    '''
    latlons -- a (N, 2) (numpy) array of latlons
    By GIS and netcdf as well as sklearn convention
    [X, 0] = lat
    [X, 1] = lon
    aka [LAT, LON] [Y,X] NOT [X,Y]!!!!!

    latlons_in_rads -- boolean stating if latlons are in radians 
    (default False -- input are in degrees)
    
    latlon0 - a (lat, lon) in degree tuple stating 
    the central point of Transverse Mercator for reprojecting to
    Northing East

    returns a (N, 2) numpy array of Northing Easting [km]
    '''
    if latlons_in_rads:
        latlons2 = np.rad2deg(latlons)
    else:
        latlons2 = latlons.copy()
    df0 = pd.DataFrame({'lat': latlons2[:, 0],
                        'lon': latlons2[:, 1]})
    pt0 = df0.apply(lambda row: Point([row.lon, row.lat]), axis=1)
    df0 = gpd.GeoDataFrame(df0, geometry=pt0, crs="EPSG:4326")
    proj4 = '+proj=tmerc +lat_0='+str(latlon0[0])
    proj4 += ' +lon_0='+str(latlon0[1])
    proj4 += ' +k=0.9996 +x_0=0 +y_0=0 +units=km'
    df1 = (gpd.GeoDataFrame(df0,
                            crs="EPSG:4326",
                            geometry=gpd.points_from_xy(df0["lon"], df0["lat"])).to_crs(proj4))
    df1['easting'] = df1.geometry.x
    df1['northing'] = df1.geometry.y
    pos = df1[["northing", "easting"]].to_numpy()
    return pos


def paired_vector_dist(yx):
    '''
    Input:
    (N, 2) array
    [X, 0] = lat or northing
    [X, 1] = lon or easting
    '''
    return yx[:, None, :] - yx


def Ls2sigma(Lx, Ly, theta):
    '''
    Lx, Ly - anistropic variogram length scales
    theta - angle relative to lines of constant latitude
    theta should be radians, and the fitting code outputs radians by default
    '''
    R = Rotation.from_rotvec([0, 0, theta])
    R = R.as_matrix()[:2, :2]
    L = np.diag([Lx**2.0, Ly**2.0])
    sigma = R @ L @ R.T
    return sigma


def compute_tau(dE, dN, sigma):
    '''
    Eq.15 in Karspeck paper
    '''
    dx_vec = np.array([dE, dN])
    return np.sqrt(dx_vec.T @ np.linalg.inv(sigma) @ dx_vec)


def compute_tau_wrapper(dyx, sigma):
    DE = dyx[:, :, 1]
    DN = dyx[:, :, 0]
    def compute_tau2(dE, dN):
        return compute_tau(dE, dN, sigma)
    compute_tau_vectorised = np.vectorize(compute_tau2)
    return compute_tau_vectorised(DE, DN)


def tau_unit_test():
    Lx = 1000.0
    Ly = 250.0
    theta = np.pi/4
    sigma = Ls2sigma(Lx, Ly, theta)
    print(sigma)
    latlon0 = (10.0, -35.0)
    # df = pd.DataFrame({'lat': [7.0, 12.0, -1.0, 20.0],
    #                    'lon': [-32.0, -24.5, -27.0, -40.0]})
    lats = np.linspace(latlon0[0]-10, latlon0[0]+10, 21)
    lons = np.linspace(latlon0[1]-10, latlon0[1]+10, 21)
    lons2, lats2 = np.meshgrid(lons, lats)
    df = pd.DataFrame({'lat': lats2.flatten().tolist(),
                       'lon': lons2.flatten().tolist()})
    print(df)
    pos = df[["lat", "lon"]].to_numpy()
    print(pos)
    pos2 = latlon2ne(pos,
                     latlons_in_rads=False,
                     latlon0=latlon0)
    print(pos2)
    df['northing'] = pos2[:, 0]
    df['easting'] = pos2[:, 1]
    paired_dis_mat = paired_vector_dist(pos2)
    print('dis_Y:')
    print(paired_dis_mat[:, :, 0])
    print('dis_X:')
    print(paired_dis_mat[:, :, 1])
    tau_mat = compute_tau_wrapper(paired_dis_mat, sigma)
    print('tau:')
    print(tau_mat)
    return {'tau': tau_mat, 'sigma': sigma, 'grid': df}


def main():
    print('=== Main ===')


if __name__ == "__main__":
    main()
