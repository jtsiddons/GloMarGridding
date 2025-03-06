'''
Fills missing pixels with HadCRUT5 defaults
Borders of infilled and noninfilled points are also adjusted
'''

import math

import iris
from iris.analysis import maths as iam
from iris.fileformats import netcdf as inc
import numpy as np
from scipy.linalg import eigh
from scipy.spatial.transform import Rotation as R

from nonstationary_cov.cube_covariance import sigma_rot_func

iris.FUTURE.save_split_attrs = True # This may become obsolete in the future

def sigma_2_parms(sigma, degrees=False):
    '''
    Takes the sigma matrix and returns Lx, Ly, theta
    
    Remind myself of the linear algebra:

    sigma = R @ diag(Lx**2,Ly**2) @ R.T
    R.T == linalg.inv(R) (rotation matricies are orthogonal)
    Hence R is made with the eigenvectors of sigma and 
    Lx**2, Ly**2 are the eigenvalues to sigma

    Note: scipy.linalg.eigh may not return eigenvalues in a nice order
    '''
    w, v = eigh(sigma)
    w_order = w.argsort()[::-1]
    w = w[w_order]
    v = v[:, w_order]
    Lx = np.sqrt(w)[0].astype(np.float32)
    Ly = np.sqrt(w)[1].astype(np.float32)
    if math.isclose(Lx, Ly, rel_tol=0.01):
        return (Lx, Ly, 0.0)
    v_padded = np.pad(v, ((0,1), (0,1)))
    v_padded[2,2] = 1
    r = R.from_matrix(v_padded)
    theta = r.as_euler('zxy', degrees=degrees)[0]
    d180 = np.pi if not degrees else 180.0
    if theta < -d180/2:
        theta = theta+d180
    if theta > d180/2:
        theta = theta-d180
    return (Lx, Ly, theta)


def zero_360_2_180_180(cube):
    '''
    Convert longitude from 0 to 360 to -180 to 180
    '''
    cube.coord('longitude').bounds = None
    cube.coord('latitude').bounds = None
    b_t_d = iris.Constraint(longitude=lambda val: val >= 180.0)
    z_2_d = iris.Constraint(longitude=lambda val: val < 180.0)
    cube_b_t_d = cube.extract(b_t_d)
    cube_z_2_d = cube.extract(z_2_d)
    new_cube_b_t_d_lons = cube_b_t_d.coord('longitude').points - 360.0
    cube_b_t_d.coord('longitude').points = new_cube_b_t_d_lons
    ans = iris.cube.CubeList()
    ans.append(cube_b_t_d)
    ans.append(cube_z_2_d)
    ans = ans.concatenate_cube()
    ans.coord('longitude').guess_bounds()
    ans.coord('latitude').guess_bounds()
    return ans


def Minus180_180_2_0_360(cube):
    '''
    Convert longitude from -180 to 180 to 0 to 360
    '''
    cube.coord('longitude').bounds = None
    cube.coord('latitude').bounds = None
    z_2_d = iris.Constraint(longitude=lambda val: val >= 0.0)
    b_t_d = iris.Constraint(longitude=lambda val: val < 0.0)
    cube_b_t_d = cube.extract(b_t_d)
    cube_z_2_d = cube.extract(z_2_d)
    new_cube_b_t_d_lons = cube_b_t_d.coord('longitude').points + 360.0
    cube_b_t_d.coord('longitude').points = new_cube_b_t_d_lons
    ans = iris.cube.CubeList()
    ans.append(cube_b_t_d)
    ans.append(cube_z_2_d)
    ans = ans.concatenate_cube()
    ans.coord('longitude').guess_bounds()
    ans.coord('latitude').guess_bounds()
    return ans


def as65_to_hs93_effective_range(L):
    '''
    Handcock-Stein 1993 and GSTAT (for common choices to v): 
    Range parameter is scaled by 1/2
    i.e.: 2 SQRT(v) r/L
    
    Abramowitz-Stegun 1965/Rasmussen-Williams 2005: 
    Range parameter is scaled by 1/SQRT(2)
    i.e.: SQRT(2 v) r/L 

    This converts L definition used in AS65/RW05 to the one used in HS93 and GSTAT (for 10 >= v >= 0.5)
    '''
    return L * 2.0/np.sqrt(2)


land_cat = {'water': 1, 'land': 2, 'coast': 16}
land_cat_inv = {v: k for k, v in land_cat.items()}
hadcrut5_defaults = {'land': {'Lx': as65_to_hs93_effective_range(1300.0),
                              'Ly': as65_to_hs93_effective_range(1300.0),
                              'sdev': 1.2,
                              'theta': 0.0},
                     'water': {'Lx': as65_to_hs93_effective_range(1300.0),
                               'Ly': as65_to_hs93_effective_range(1300.0),
                               'sdev': 0.6,
                               'theta': 0.0}}


def landfrac_categorial(land_frac_arr,
                        minimum_land_threshold=0.001):
    '''
    Create a uint8 ndarray based on land fraction
    uint8 flags based on convention used in OSTIA

    OSTIA definition
    flag_masks        array([ 1, 2, 4, 8, 16], dtype=int8)
    flag_meanings     'water land optional_lake_surface sea_ice optional_river_surface' 
    '''
    landfrac_categorial_arr = np.zeros_like(land_frac_arr, dtype=np.uint8)
    water = land_frac_arr <= minimum_land_threshold
    land  = land_frac_arr >= (1.0-minimum_land_threshold)
    coast = np.logical_and(land_frac_arr > minimum_land_threshold,
                           land_frac_arr < (1.0-minimum_land_threshold))
    landfrac_categorial_arr[water] = land_cat['water']
    landfrac_categorial_arr[land] = land_cat['land']
    landfrac_categorial_arr[coast] = land_cat['coast']
    return landfrac_categorial_arr


def ESA_landfrac(minimum_land_threshold=0.001,
                 convert2_0_360=False):
    '''
    Make ESA land fraction data cube

    ESA land (sea) fraction:
    Have no lakes, but lakes are included in HadCRUT5 weight files and ERA5. However for temperature kriging,
    it is not clear if lakes should be treated as land or even open sea, so defaults using for sea or land
    or even interpolating from nearby grid point (mostly land) be valid. 
    ERA LSAT kriging scales exclude lakes because lakes are labeled as water pixels.
    Coverage is globally complete; there are no missing data pixels (thumbs up).
    Cube is from -180 to +180 deg longitude
    '''
    in_path = '/gws/nopw/j04/hostace/data/ESA_CCI_5deg_monthly_extra/ANOMALY/'
    in_file = in_path+'19800115_regridded_sst.nc'
    sea_frac = iris.load_cube(in_file, 'sea_area_fraction')[0] # NOT land fraction
    land_frac = iam.multiply(iam.add(sea_frac, -1), -1)
    if convert2_0_360:
        land_frac = Minus180_180_2_0_360(land_frac)
    land_frac.rename('land area fraction')
    land_frac_cat = land_frac.copy()
    land_frac_cat.data = landfrac_categorial(land_frac_cat.data,
                                             minimum_land_threshold=minimum_land_threshold)
    land_frac_cat.rename('land area categorical')
    land_frac_cat.attributes = None
    land_frac_cat.attributes['definitions'] = 'water = 1, land = 2, coast = 16'
    ans = iris.cube.CubeList()
    ans.append(land_frac)
    ans.append(land_frac_cat)
    return ans


# def HadCRUT5_landfrac(minimum_land_threshold=0.25):
#     '''
#     In HadCURT5 analysis weight file, any pixels that have a shred of possibility to be
#     land (or sea ice) will have a non-0 value along the axis=0.
#     Permenant land always have 1.0
#     Permenant sea always have 0.0 at all times along axis=0.0
#     Having missing data pixels :(.
#     From -180 to +180 deg longitude

#     /gws/nopw/j04/hostace/data/HadCRUT5
#     '''
#     return

def find_neighbours(jj, ii, jj_max=35, ii_max=71):
    '''
    Find immediate neighbouring grid points,
    accounting for wrap around point

    Parameters
    ----------
    jj, ii : int
        lat/y and lon/x
    jj_max, ii_max: int
        Number of lat and lon points
    Returns
    -------
    list of tuples of (lat_index, lon_index) of the neighbours    
    '''
    assert 0 <= jj <= jj_max, 'jj is out of range: jj='+str(jj)
    assert 0 <= ii <= ii_max, 'ii is out of range: ii='+str(ii)
    neighbours = []
    for jj_s in [jj-1, jj, jj+1]:
        if (jj_s < 0) or (jj_s > jj_max):
            continue
        for ii_s in [ii-1, ii, ii+1]:
            if ii_s > ii_max:
                neighbours.append((jj_s, 0))
            elif ii_s < 0:
                neighbours.append((jj_s, ii_max))
            else:
                neighbours.append((jj_s, ii_s))
    return neighbours


def infill_scales(Lx, Ly,
                  theta,
                  sdev,
                  convergence_qc,
                  terrain='land',
                  infill_parms_lookup=None,
                  landfrac_func=ESA_landfrac,
                  landfrac_func_kwargs=None,
                  fudge_long_scales=True):
    '''
    Fill Matern kernel parameters into grid points that are:
    - masked
    - where the kernel parameters have failed to converge
    
    "infill_parms_lookup" must be a dict with keys that work with "terrain"

    Parameters
    ----------
    Lx, Ly, theta, sdev : iris.cube.Cube
        iris cubes with Lx, Ly, theta, sdev with gaps to fill in
    convergence_qc : iris.cube.Cube
        iris cubes where QC/convergence flags are stored
    terrain : str
        terrain type, must be a key to infill_parms_lookup
    infill_parms_lookup : dict
        look up table for priors to the infill parameters
        see hadcrut5_defaults for format
    landfrac_func : callable
        Function that returns the land fraction
    landfrac_func_kwargs : dict
        kwargs for landfrac_func
    fudge_long_scales : bool
        Should Lx, Ly, theta, sdev be modified if convergence_qc values that are flagged
        qc_redflags = [2.0, 3.0, 9.0]
        See cube_covarinace.py
        2.0 : success but with one parameter reaching upper bounadries
        3.0 : success with multiple parameters reaching the boundaries
        9.0 : No convergence after scipy.optimize.minimize reach maxiter
    '''
    #
    if infill_parms_lookup is None:
        infill_parms_lookup = hadcrut5_defaults
    if landfrac_func_kwargs is None:
        landfrac_func_kwargs = {}
    #
    assert terrain in infill_parms_lookup, 'Unknown terrain; land or sea only'
    infill_parms = infill_parms_lookup[terrain]
    #
    where_is_mask = Lx.data.mask.copy()
    Lx_v2 = Lx.copy()
    Ly_v2 = Ly.copy()
    theta_v2 = theta.copy()
    sdev_v2 = sdev.copy()
    #
    Lx_v2.data.mask = False
    Ly_v2.data.mask = False
    theta_v2.data.mask = False
    sdev_v2.data.mask = False
    #
    Lx_v2.data[where_is_mask] = infill_parms['Lx']
    Ly_v2.data[where_is_mask] = infill_parms['Ly']
    theta_v2.data[where_is_mask] = infill_parms['theta']
    sdev_v2.data[where_is_mask] = infill_parms['sdev']
    #
    if fudge_long_scales:
        qc_redflags = [2.0, 3.0, 9.0]
        where_are_redflags = np.isin(convergence_qc.data, qc_redflags)
        if np.sum(where_are_redflags) > 0:
            print('Red flags detected, infilling: '+str(np.sum(where_are_redflags)))
            Lx_v2.data[where_are_redflags] = infill_parms['Lx']
            Ly_v2.data[where_are_redflags] = infill_parms['Ly']
            theta_v2.data[where_are_redflags] = infill_parms['theta']
            sdev_v2.data[where_are_redflags] = infill_parms['sdev']
    #
    Lx_v3 = Lx_v2.copy()
    Ly_v3 = Ly_v2.copy()
    theta_v3 = theta_v2.copy()
    sdev_v3 = sdev_v2.copy()
    #
    out_cubelist = iris.cube.CubeList()
    #
    land_info = landfrac_func(**landfrac_func_kwargs)
    land_tac = land_info.extract('land area categorical')[0]
    where_is_coast = land_tac.data == land_cat['coast']
    where_is_current_terr = land_tac.data == land_cat[terrain]
    #
    where_is_correct_terr = np.logical_or(where_is_coast, where_is_current_terr)
    where_is_mask_and_need_avging = np.logical_and(where_is_mask, where_is_correct_terr)
    where_is_mask_and_not_avged = np.logical_and(where_is_mask, np.logical_not(where_is_correct_terr))
    #
    infill_status = land_tac.copy()
    infill_status.data[:] = 0
    infill_status.data[where_is_mask_and_not_avged] = 1
    infill_status.data[where_is_mask_and_need_avging] = 2
    infill_status.data[where_are_redflags] = 3
    infill_status.rename('infill status')
    infill_status.attributes = None
    definition_str = 'not_adj=0 '
    definition_str += 'adj_2_hadcrut5_parms=1 '
    definition_str += 'adj_2_bavg_with_hadcrut5_parms_prior=2 '
    definition_str += 'adj_2_bavg_with_hadcrut5_parms_prior_due_to_unconvergence=3'
    infill_status.attributes['definitions'] = definition_str
    #
    where_2_bavg_list = [where_is_mask_and_need_avging]
    if fudge_long_scales:
        where_2_bavg_list.append(where_are_redflags)
    for where_2_bavg in where_2_bavg_list:
        jjs, iis = np.where(where_2_bavg)
        if iis.size > 0:
            for jj, ii in zip(jjs, iis):
                neighbours = find_neighbours(jj, ii)
                Lxs, Lys, thetas, sdevs = [], [], [], []
                sigmas = []
                for nei in neighbours:
                    Lxs.append(Lx_v2.data[nei[0], nei[1]])
                    Lys.append(Ly_v2.data[nei[0], nei[1]])
                    thetas.append(theta_v2.data[nei[0], nei[1]])
                    sdevs.append(sdev_v2.data[nei[0], nei[1]])
                Lxs = np.array(Lxs)
                Lys = np.array(Lys)
                thetas = np.array(thetas)
                sdevs = np.array(sdevs)
                for xL, yL, ateht in zip(Lxs, Lys, thetas):
                    sigmas.append(sigma_rot_func(xL, yL, ateht))
                sigmas = np.stack(sigmas)
                sigma_bar = np.average(sigmas, axis=0)
                xL_new, yL_new, ateht_new = sigma_2_parms(sigma_bar)
                Lx_v3.data[jj, ii] = xL_new
                Ly_v3.data[jj, ii] = yL_new
                theta_v3.data[jj, ii] = ateht_new
                sdev_v3.data[jj, ii] = np.sqrt(np.mean(np.square(sdevs)))
    out_cubelist.append(Lx_v3)
    out_cubelist.append(Ly_v3)
    out_cubelist.append(theta_v3)
    out_cubelist.append(sdev_v3)
    out_cubelist.append(land_tac)
    out_cubelist.append(infill_status)
    return out_cubelist


def run_land():
    ''' Fill the land scale files '''
    terrain = 'land'
    basepath = '/noc/mpoc/surface/ERA5_SURFTEMP_500deg_monthly/ANOMALY/SpatialScales/'
    inpath = basepath+'matern_physical_distances_v_eq_1p5/'
    outpath = basepath+'matern_physical_distances_v_eq_1p5_filled_in/'
    for month in [m+1 for m in range(12)]:
        ncfile = 'lsat_'+str(month).zfill(2)+'.nc'
        cubes = iris.load(inpath+ncfile)
        Lx = cubes.extract('Lx')[0]
        Ly = cubes.extract('Ly')[0]
        theta = cubes.extract('theta')[0]
        sdev = cubes.extract('standard_deviation')[0]
        qc = cubes.extract('qc_code')[0]
        ans = infill_scales(Lx, Ly,
                            theta,
                            sdev,
                            qc,
                            terrain=terrain)
        print(ans)
        print('Saving to '+outpath+ncfile)
        inc.save(ans, outpath+ncfile)


def run_sea():
    ''' Fill the sea scale files '''
    terrain = 'water'
    basepath = '/noc/mpoc/surface_data/ESA_CCI5deg_month_extra/ANOMALY/SpatialScales/'
    inpath = basepath+'matern_physical_distances_v_eq_1p5/'
    outpath = basepath+'matern_physical_distances_v_eq_1p5_filled_in/'
    for month in [m+1 for m in range(12)]:
        ncfile = 'sst_'+str(month).zfill(2)+'.nc'
        cubes = iris.load(inpath+ncfile)
        Lx = cubes.extract('Lx')[0]
        Ly = cubes.extract('Ly')[0]
        theta = cubes.extract('theta')[0]
        sdev = cubes.extract('standard_deviation')[0]
        qc = cubes.extract('qc_code')[0]
        ans = infill_scales(Lx, Ly,
                            theta,
                            sdev,
                            qc,
                            terrain=terrain)
        print(ans)
        print('Saving to '+outpath+ncfile)
        inc.save(ans, outpath+ncfile)


def main():
    ''' Main - keep calm and does something! '''
    print('--- MAIN ---')
    run_land()
    run_sea()


if __name__ == "__main__":
    main()
