'''
Requires numpy, scipy, sklearn
iris needs to be installed (it is required by other modules within this package
xarray cubes should work via iris interface
'''
import datetime
from functools import reduce
import numbers
import sys
import tracemalloc

from joblib import Parallel, delayed  # Developmental
# from iris import analysis as ia
import numpy as np
from numpy import ma
from numpy import linalg
from scipy.special import kv as modified_bessel_2nd
from scipy.special import gamma
from statsmodels.stats import correlation_tools

from ellipse_estimation import cube_covariance as cube_cov
from ellipse_estimation.distance_util import scalar_cube_great_circle_distance

# Below is in theory redudant, but the view/controller bits of the code has not been
# integrated to the package; for now, keeping this in case of breaking other code
# from ellipse_estimation.distance_util import scalar_cube_great_circle_distance_cube

# Developmental functions that I do not have time to explore much
_default_n_jobs = 4
# This is the only one would work if you want to modify self covariance
# array inside method; slow
# _default_backend = 'threading'
# Method is modified to return a list of numbers, hence not restricted to
_default_backend = 'loky'

# _MAX_DEG_Kar = 20.0  # Karspeck et al distance threshold in degrees latlon
# _MAX_DIST_Kar = cube_cov._deg2km(_MAX_DEG_Kar)  # to km @ lat = 0.0 (2222km)

# _MAX_DIST_UKMO = 10000.0  # UKMO uses 10000km range to fit the non-rot ellipse
# _MAX_DEG_UKMO = cube_cov._km2deg(_MAX_DIST_UKMO)

_MAX_DIST_compromise = 6000.0  # Compromise _MAX_DIST_Kar &_MAX_DIST_UKMO
# _MAX_DEG_compromise = cube_cov._km2deg(_MAX_DIST_compromise)

# _MIN_CORR_Threshold = 0.5 / np.e


def convert_cube_data_2_MaskedArray_if_not(cube):
    '''
    Forces cube.data to be an instance of np.ma.MaskedArray

    Parameters
    ----------
    cube : instance to iris.cube.Cube
        iris.cube.Cube.data can be masked or not masked

    Returns
    -------
    cube : instance to iris.cube.Cube
        Perhaps the data within the cube is now an instance of np.ma.MaskedArray
    '''
    np_array = cube.data
    print(type(np_array))
    assert isinstance(
        np_array, np.ndarray), 'Not a numpy array (masked or not)'
    if not isinstance(np_array, np.ma.MaskedArray):
        print('Ad hoc conversion to np.ma.MaskedArray')
        cube.data = np.ma.MaskedArray(np_array)
        return cube
    return cube


def sizeof_fmt(num, suffix="B"):
    '''
    Convert numbers to kilo/mega... bytes,
    for interactive printing of code progress
    '''
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def c_ij_anistropic_rotated_nonstationary(v,
                                          sdev_i, sdev_j,
                                          x_i, x_j,
                                          sigma_parms_i, sigma_parms_j,
                                          verbose=False):
    '''
    Compute the nonstationary spatially-varying covariance between
    point i and j with covariance model parameters sigma_parms_i, sigma_parms_j
    and local standard deviations of sdev_i, sdev_j

    x_i = zonal displacement (NOT COORDINATES OF point i)
    x_j = meridonal displacement (NOT COORDINATES OF point j)
    Use scalar_cube_great_circle_distance to compute displacement and distance

    ans of scalar_cube_great_circle_distance: x_i = ans[2] and x_j = ans[1]
    (ans[0] is the great circle dist)

    sigma_parms_i = (Lx_i, Ly_i, theta_i)
    sigma_parms_j = (Lx_j, Ly_j, theta_j)

    original equation:
    1) Paciorek and Schevrish 2006 Equation 8 https://doi.org/10.1002/env.785
    2) Karspeck et al 2012 Equation 17 https://doi.org/10.1002/qj.900

    Parameters
    ----------
    v : float
        Matern shape parameter
    sdev_i, sdev_j: float
        Standard deviations of the two points
    x_i, x_j: float
        Components of the vector displacement between the two points
    sigma_parms_i, sigma_parms_j: iterable
        iterable with a length of 3 each, that states Lx, Ly, theta
        Dimensions of Lx, Ly, x_i and x_j must be the same (i.e. km with km
        not km with degrees)
    verbose=False: bool
        More stuff gets printed out to stdout!

    Returns
    -------
    ans : float
        the covariance between the two points
    '''
    #
    # Compute sigma_bar
    sigma_i = cube_cov.sigma_rot_func(sigma_parms_i[0],
                                      sigma_parms_i[1],
                                      sigma_parms_i[2])
    sigma_j = cube_cov.sigma_rot_func(sigma_parms_j[0],
                                      sigma_parms_j[1],
                                      sigma_parms_j[2])
    sigma_bar = 0.5 * (sigma_i + sigma_j)
    if verbose:
        print('sigma_bar = ', sigma_bar)
    #
    # sigma_bar can be broken down to new sigma parameters
    # aka a new Lx, Ly and theta using eigenvalue decomposition
    # Sigma_bar = R(theta_bar) @ [[Lx_bar**2 0 ][0 Ly_bar**2]] @ R(theta_bar)_inverse
    # In which eigenvalues to Sigma Bar forms the diagonal matrix of Lx_bar Ly_bar
    # and eigenvectors are rotation matrix R(theta_bar)
    # if sigma_bar is nearly circle,
    # there are a possibility of floating point issue
    # i.e. eigenvalues and eigenvectors become "complex"
    # when off-diagonal are a very small float (1E-10)
    #
    near_zero_check = np.isclose(sigma_bar, 0.0)
    if np.any(near_zero_check):
        # This gets annoying if running istropic unit tests...
        # print('Adjustments made to small off diagonal terms of sigma_bar')
        sigma_bar[near_zero_check] = 0.0
    #
    if verbose:
        # Decomposing sigma_bar as an optional verbose
        # sigma_bar = R_bar x [( Lx_bar**2 0 ) (0 Ly_bar**2)] x R_bar_transpose
        sigma_bar_eigval, sigma_bar_eigvec = linalg.eig(sigma_bar)
        #
        # If numerical instability is detected, resulting in complex number, take the real part
        try:
            assert np.all(np.isreal(sigma_bar_eigval)) and np.all(
                np.isreal(sigma_bar_eigvec))
        except AssertionError:
            print('Complex eigenvalues detected!')
            print('sigma_bar_eigval = ', sigma_bar_eigval)
            print('sigma_bar_eigvec = ', sigma_bar_eigvec)
            sigma_bar_eigval = sigma_bar_eigval.real
            sigma_bar_eigvec = sigma_bar_eigvec.real
        # Actual Lx, Ly are square roots of the eigenvalues
        Lx_bar = np.sqrt(sigma_bar_eigval[0])
        Ly_bar = np.sqrt(sigma_bar_eigval[1])
        #
        # https://stackoverflow.com/questions/15022630/how-to-calculate-the-angle-from-rotation-matrix
        #
        print(f'Eigvals of sigma_bar = {sigma_bar_eigval}')
        print(f"(Lx_bar, Ly_bar) = ({Lx_bar},{Ly_bar})")
        print(f'Eigvec of sigma_bar  = {sigma_bar_eigvec}')
        # Use arctan2 to compute rotation angle '''
        theta_bar = np.arctan2(sigma_bar_eigvec[1, 0], sigma_bar_eigvec[0, 0])
        # Below should show the same angle in radians
        v_ans0 = (np.arccos(sigma_bar_eigvec[0, 0]),
                  np.rad2deg(np.arccos(sigma_bar_eigvec[0, 0])))
        v_ans1 = (np.arcsin(sigma_bar_eigvec[0, 1]),
                  np.rad2deg(np.arcsin(sigma_bar_eigvec[0, 1])))
        v_ans2 = (-np.arcsin(sigma_bar_eigvec[1, 0]),
                  np.rad2deg(-np.arcsin(sigma_bar_eigvec[1, 0])))
        v_ans3 = (theta_bar, np.rad2deg(theta_bar))
        print(f'arccos  R[0,0][1, 1] = {v_ans0} (>0 for ang within +/- pi/2)')
        print(f'arcsin  R[0,1]       = {v_ans1}')
        print(f'-arccos R[1,0]       = {v_ans2}')
        print(f'theta_bar            = {v_ans3} (using arctan2)')
        tau_bar = cube_cov.mahal_dist_func_rot(x_i, x_j,
                                               Lx_bar, Ly_bar,
                                               theta=theta_bar,
                                               verbose=verbose)
    else:
        # Direct computation without decomposition (faster)
        # This is direct use of right part of Equation 18 in Karspeck et al 2012
        # This is behind else, so one won't be computing stuff twice
        tau_bar = cube_cov.mahal_dist_func_sigma(x_i, x_j,
                                                 sigma_bar,
                                                 verbose=verbose)
    if verbose:
        print('xi, xj  = ', x_i, x_j)
        print('tau_bar = ', tau_bar)
    #
    # Eq 17 in Karspeck et al 2012
    # ans = first_term x second_term x third_term x fourth_term '''
    first_term = (sdev_i * sdev_j)/(gamma(v) * (2.0**(v-1)))

    def root4(val):
        ''' 4th root '''
        return np.sqrt(np.sqrt(val))

    def det22(m22):
        ''' Explict computation of determinant of 2x2 matrix '''
        return m22[0, 0]*m22[1, 1]-m22[0, 1]*m22[1, 0]
    # second_term_u = root4(linalg.det(sigma_i))*root4(linalg.det(sigma_j))
    # second_term_d = np.sqrt(linalg.det(sigma_bar))
    second_term_u = root4(det22(sigma_i))*root4(det22(sigma_j))
    second_term_d = np.sqrt(det22(sigma_bar))
    second_term = second_term_u/second_term_d
    third_term = (2.0 * tau_bar * np.sqrt(v))**v
    forth_term = modified_bessel_2nd(v, 2.0 * tau_bar * np.sqrt(v))
    ##
    if verbose:
        print('Check: first_term  = ', first_term)
        print('Check: second_term = ', second_term)
        print('Check: third_term  = ', third_term)
        print('Check: forth_term  = ', forth_term)
    ##
    ans = first_term * second_term * third_term * forth_term
    if verbose:
        print('Check: sdev_i * sdev_j = ', sdev_i * sdev_j)
        print('Check: ans (cov)       = ', ans)
        print('Check: ans (cor)       = ', ans/(sdev_i * sdev_j))
    # Don't know why I added the below, I have never seen it triggered...
    # in which you are not supposed to as it will indicate bug elsewhere
    # the original check is much more
    if ans > (sdev_i * sdev_j):
        raise ValueError('sdev_i * sdev_j should always be smaller than ans')
    return ans


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    ''' Helper function for perturb_sym_matrix_2_positive_definite '''
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def perturb_sym_matrix_2_positive_definite(square_sym_matrix):
    '''
    On the fly eigenvalue clipping, this is based statsmodels code
    statsmodels.stats.correlation_tools.cov_nearest
    statsmodels.stats.correlation_tools.corr_nearest

    Use repair_damaged_covariance instead, it is more complete

    Other methods exist:
    https://nhigham.com/2021/02/16/diagonally-perturbing-a-symmetric-matrix-to-make-it-positive-definite/
    https://nhigham.com/2013/02/13/the-nearest-correlation-matrix/
    https://academic.oup.com/imajna/article/22/3/329/708688
    '''
    matrix_dim = square_sym_matrix.shape
    if (len(matrix_dim) != 2) or (matrix_dim[0] != matrix_dim[1]) or not check_symmetric(square_sym_matrix):
        raise ValueError('Matrix is not square and/or symmetric.')
    ##
    eigenvalues = linalg.eigvalsh(square_sym_matrix)
    min_eigen = np.min(eigenvalues)
    max_eigen = np.max(eigenvalues)
    n_negatives = np.sum(eigenvalues < 0.0)
    print('Number of eigenvalues = ', len(eigenvalues))
    print('Number of negative eigenvalues = ', n_negatives)
    print('Largest eigenvalue  = ', max_eigen)
    print('Smallest eigenvalue = ', min_eigen)
    if min_eigen >= 0.0:
        print('Matrix is already positive (semi-)definite.')
        return square_sym_matrix
    ans = correlation_tools.cov_nearest(square_sym_matrix)
    #
    eigenvalues_adj = linalg.eigvalsh(ans)
    min_eigen_adj = np.min(eigenvalues_adj)
    max_eigen_adj = np.max(eigenvalues_adj)
    n_negatives_adj = np.sum(eigenvalues_adj < 0.0)
    print('Post adjustments:')
    print('Number of negative eigenvalues (post_adj) = ', n_negatives_adj)
    print('Largest eigenvalue (post_adj)  = ', max_eigen_adj)
    print('Smallest eigenvalue (post_adj) = ', min_eigen_adj)
    return ans


# def seaice_anti_hubris_field(cube, land_mask, ice_fill_value=None):
#     '''
#     We have other ways to fill in data gaps now,
#     like using HadCRUT5 parameters...
#     but this function can be resurrected in the future
#
#     assuming sea == 1 and land == 0 in land_mask
#     replace where xor(land_mask, original_cube) (aka masked sea points)
#     with ice_fill value
#     default for ice_fill_value are applicable
#
#     WARNING:
#     Inserting same values over multiple rows and columns
#     lead to degenerate matrices!!!
#
#     Possible solution:
#     Instead of simple infilling, add a random pertubation on top
#     '''
#     cube2 = cube.copy()
#     cube2.data.mask = False
#     cube2.data = np.ma.masked_where(land_mask.data < 0.95, cube2.data)
#     where_are_the_ice = np.logical_xor(cube.data.mask, cube2.data.mask)
#     if ice_fill_value is None:
#         if cube.units == 'km':
#             ice_fill_value = 100.0
#         elif cube.units == 'radians':
#             ice_fill_value = 0.0
#         else:
#             err_msg = 'ice_fill_value not provided, no defaults for cube units'
#             raise ValueError(err_msg)
#     ##
#     if isinstance(ice_fill_value, numbers.Number):
#         print('Replacing all xor(land_mask, cube) points with ', ice_fill_value)
#         cube2.data[where_are_the_ice] = ice_fill_value
#         return cube2
#     elif np.ndim(ice_fill_value) != 0:
#         ''' This allow a fillin by user-provided vector/list, length needs to match '''
#         ice_fill_value2 = np.array(ice_fill_value)
#         ice_fill_value2 = ice_fill_value2[:, np.newaxis]
#         where_are_the_ice2 = where_are_the_ice.astype(float)
#         fill_in_matrix = np.multiply(ice_fill_value2, where_are_the_ice2)
#         cube2.data[where_are_the_ice] = 0.0
#         cube2.data = cube2.data + fill_in_matrix
#         print('Replacing all xor(land_mask, cube) points with ',ice_fill_value[0],' ... ', ice_fill_value[-1])
#         return cube2
#     elif isinstance(ice_fill_value, str):
#         '''
#         Variations of mean sub in imputation
#         for distances imputation by minimum (anti-hubris value)
#         for angles, "minimum" approach would be unrealistic, so we can use median (or angular mean, but that isn't implemented in iris)
#         '''
#         if ice_fill_value == 'zonal_min_substitution':
#             zonal_mean = cube.collapsed('longitude', ia.MIN)
#             zonal_mean_val = zonal_mean.data
#             zonal_mean_val = zonal_mean_val[:, np.newaxis]
#             cube2.data[where_are_the_ice] = 0.0
#             fill_in_matrix = np.multiply(zonal_mean_val, where_are_the_ice.astype(float))
#             cube2.data = cube2.data + fill_in_matrix
#             print('Replacing all xor(land_mask, cube) points with ',zonal_mean.data[0],' ... ', zonal_mean.data[-1])
#             return cube2
#         elif ice_fill_value == 'zonal_median_substitution':
#             zonal_mean = cube.collapsed('longitude', ia.MEDIAN)
#             zonal_mean_val = zonal_mean.data
#             zonal_mean_val = zonal_mean_val[:, np.newaxis]
#             cube2.data[where_are_the_ice] = 0.0
#             fill_in_matrix = np.multiply(zonal_mean_val, where_are_the_ice.astype(float))
#             cube2.data = cube2.data + fill_in_matrix
#             print('Replacing all xor(land_mask, cube) points with ', zonal_mean.data[0], ' ... ', zonal_mean.data[-1])
#             return cube2
#         elif ice_fill_value == 'zonal_mean_substitution':
#             zonal_mean = cube.collapsed('longitude', ia.MEAN)
#             zonal_mean_val = zonal_mean.data
#             zonal_mean_val = zonal_mean_val[:, np.newaxis]
#             cube2.data[where_are_the_ice] = 0.0
#             fill_in_matrix = np.multiply(zonal_mean_val,
#                                          where_are_the_ice.astype(float))
#             cube2.data = cube2.data + fill_in_matrix
#             print('Replacing all xor(land_mask, cube) points with ',
#                   zonal_mean.data[0],
#                   ' ... ',
#                   zonal_mean.data[-1])
#             return cube2
#         else:
#             raise ValueError('Unknown string input for ice_fill_value')
#     else:
#         raise ValueError('Unknown input for ice_fill_value')


class CovarianceCube_PreStichedLocalEstimates():

    '''
    The class that takes multiple iris cubes of 
    non-stationary variogram parameters to build
    and save covariance matrices
    '''

    def __init__(self,
                 Lx_cube, Ly_cube,
                 theta_cube,
                 sdev_cube,
                 v=3,
                 output_floatprecision=np.float64,
                 max_dist=_MAX_DIST_compromise,
                 degree_dist=False,
                 delta_x_method="Modified_Met_Office",
                 check_positive_definite=False,
                 use_joblib=False,
                 n_jobs=_default_n_jobs,
                 backend=_default_backend,
                 nolazy=False,
                 use_sklearn_haversine=False,
                 verbose=False):
        '''
        v = Matern covariance shape parameter

        As not shown on TV (Karspeck et al)...
        Lx - an iris (or xarray2iris-converted?) cube of horizontal length scales (
        Ly - an iris (or xarray2iris-converted?) cube of meridonal length scales
        theta - an iris cube of rotation angles (RADIANS ONLY)

        sdev - standard deviation -- right now it just takes a number cube
        if you have multiple contribution to sdev (uncertainities derived from different sources), you need
        to put them into one cube

        Rules:
        Valid (ocean) point:
        1) cov_ns and cor_ns are computed out to max_dist; out of range = 0.0
        2) Masked points are ignored

        Invalid (masked) points:
        1) Skipped over

        max_dist:
        float (km) or (degrees if you want to work in degrees), default 6000km
        if you want infinite distance, just set it to some stupidly large number,
        fun numbers to use:
        1.5E8 (i.e. ~1 astronomical unit (Earth-Sun distance)
        5.0E9 (average distance between Earth and not-a-planet-anymore Pluto

        Parameters
        ----------
        Lx_cube, Ly_cube, theta_cube, sdev_cube: instances to iris.cube.Cube
            cubes with non-stationary parameters
        v=3: float
            Matern shape parameter
        output_floatprecision :
            Float point precision of the ouput covariance
            numpy defaults to float64, 
            noting that float32 halves the storage and halves the memory to use
        max_dist : float
            If the Haversine distance between 2 points exceed max_dist,
            covariance is set to 0
        degree_dist : bool
            Distances are based on degrees
        delta_x_method : str
            How are displacements computed between points
            The default is the same as in cube_covariance "Modified_Met_Office"

        check_positive_definite : bool
            For production this should be False
            but for unit testing it should be True,
            if True a quick on the fly eigenvalue clipping
            will be conducted, if constructed covariance is not
            positive (semi)definite.

        use_joblib : bool
            Should joblib parallel processing be used

        n_jobs : int
            Number of parallel thread, only matter of use_joblib is
            true. Otherwise numpy will uses its own parallelisation

        backend : str
            backend of joblib

        nolazy : bool
            Manually forces computation to occur

        use_sklearn_haversine: bool
            sklearn has haversine function, but its preformance
            is inconsistent between different machines, and can
            cause a major slow down.

        verbose : bool
            More stout stuff!
        '''
        tracemalloc.start()
        ove_start_time = datetime.datetime.now()
        print('Overhead processing start: ',
              ove_start_time.strftime("%Y-%m-%d %H:%M:%S"))

        if not use_joblib:
            n_jobs = 1

        if not isinstance(max_dist, numbers.Number):
            raise ValueError('max_dist must be a number')

        # Defining the input data
        self.v = v  # Matern covariance shape parameter
        self.Lx_local_estimates = convert_cube_data_2_MaskedArray_if_not(
            Lx_cube)
        self.Ly_local_estimates = convert_cube_data_2_MaskedArray_if_not(
            Ly_cube)
        self.theta_local_estimates = convert_cube_data_2_MaskedArray_if_not(
            theta_cube)
        self.sdev_local_estimates = convert_cube_data_2_MaskedArray_if_not(
            sdev_cube)
        self.max_dist = max_dist
        self.degree_dist = degree_dist
        self.delta_x_method = delta_x_method
        self.check_positive_definite = check_positive_definite
        self.use_sklearn_haversine = use_sklearn_haversine

        # print('Fortran ordering check')
        # for selfcubes in [self.Lx_local_estimates,
        #                   self.Ly_local_estimates,
        #                   self.theta_local_estimates,
        #                   self.sdev_local_estimates]:
        #     print(repr(selfcubes), np.isfortran(selfcubes.data))

        if nolazy:
            for selfcubes in [self.Lx_local_estimates,
                              self.Ly_local_estimates,
                              self.theta_local_estimates,
                              self.sdev_local_estimates]:
                self._no_lazy_data(selfcubes)

        # The cov and corr matrix will be sq matrix of this
        self.xy_shape = self.Lx_local_estimates.shape
        self.n_elements = reduce(lambda x, y: x*y, self.xy_shape)
        self.data_has_mask = ma.is_masked(self.Lx_local_estimates.data)
        if self.data_has_mask:
            print('Masked pixels detected in input files')
            self.cube_mask = self.Lx_local_estimates.data.mask
            self.cube_mask_1D = self.cube_mask.flatten()
            self.covar_size = np.sum(np.logical_not(self.cube_mask))
        else:
            print('No masked pixels')
            self.cube_mask = np.zeros_like(self.Lx_local_estimates.data.data,
                                           dtype=bool)
            self.cube_mask_1D = self.cube_mask.flatten()
            self.covar_size = self.n_elements

        # Prepare matricies
        # i and j represent data pairs, the original point follows _i, remote point is _j
        print('Creating dummy arrays')
        self.cov_ns = np.zeros(
            (self.covar_size, self.covar_size), dtype=output_floatprecision)
        self.lat_mat_i = np.zeros(
            (self.covar_size, self.covar_size), dtype=np.float16)
        self.lon_mat_i = np.zeros(
            (self.covar_size, self.covar_size), dtype=np.float16)
        self.lat_mat_j = np.zeros(
            (self.covar_size, self.covar_size), dtype=np.float16)
        self.lon_mat_j = np.zeros(
            (self.covar_size, self.covar_size), dtype=np.float16)

        ##
        print('Compressing (masked) array to 1D')
        self.Lx_local_estimates_compressed = self.Lx_local_estimates.data.compressed()
        self.Ly_local_estimates_compressed = self.Ly_local_estimates.data.compressed()
        self.theta_local_estimates_compressed = self.theta_local_estimates.data.compressed()
        self.sdev_local_estimates_compressed = self.sdev_local_estimates.data.compressed()

        self.xx, self.yy = np.meshgrid(Lx_cube.coord('longitude').points,
                                       Lx_cube.coord('latitude').points)
        self.xm = np.ma.masked_where(self.cube_mask, self.xx)
        self.ym = np.ma.masked_where(self.cube_mask, self.yy)
        self.lat_grid_compressed = self.ym.compressed()
        self.lon_grid_compressed = self.xm.compressed()
        self.xy = np.column_stack(
            [self.lon_grid_compressed, self.lat_grid_compressed])
        self.xy_full = np.column_stack([self.xm.flatten(), self.ym.flatten()])

        ove_end_time = datetime.datetime.now()
        print('Overhead processing ended: ',
              ove_end_time.strftime("%Y-%m-%d %H:%M:%S"))
        print('Time ellipsed: ', ove_end_time-ove_start_time)

        #
        # Fill in the covariance matrix by looping over grid points with different sigma matricies
        ii_index = range(self.covar_size)
        cov_start_time = datetime.datetime.now()
        print('Covariance processing start: ',
              cov_start_time.strftime("%Y-%m-%d %H:%M:%S"))
        for ii in ii_index:
            curr, peak = tracemalloc.get_traced_memory()
            rt_row_info = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+': '
            rt_row_info += 'Row '+str(ii)+'/'+str(ii_index[-1])+'; '
            rt_row_info += f'current & peak mem: {curr/(1024*1024)} {peak/(1024*1024)} MB'
            rt_row_info += '; n_jobs = '+str(n_jobs)
            print(rt_row_info)
            self.cov_ns[ii, ii] = self.sdev_local_estimates_compressed[ii]**2.0
            self.lat_mat_i[ii, ii] = self.lat_mat_j[ii,
                                                    ii] = self.lat_grid_compressed[ii]
            self.lon_mat_i[ii, ii] = self.lon_mat_j[ii,
                                                    ii] = self.lon_grid_compressed[ii]
            jj_index = range(ii+1, self.covar_size)
            #
            # New -- moved this outside the loop, may make things slightly faster
            # (code will at least look cleaner)
            sdev_i = self.sdev_local_estimates_compressed[ii].copy()
            sigma_parms_i = (self.Lx_local_estimates_compressed[ii].copy(),
                             self.Ly_local_estimates_compressed[ii].copy(),
                             self.theta_local_estimates_compressed[ii].copy())
            lat_grid_compressed_i = self.lat_grid_compressed[ii].copy()
            lon_grid_compressed_i = self.lon_grid_compressed[ii].copy()
            if verbose:
                print(ii,
                      sdev_i,
                      sigma_parms_i,
                      lat_grid_compressed_i,
                      lon_grid_compressed_i)  # For checking use
            ##
            # Potential for improvement --
            # it should be possible to make this faster
            # as it is is embarrassingly parallel
            # Nevertheless it doesn't run slow.
            # Estimating the individual local sigma (cube_covariance.py)
            # is much slower
            #
            # interface info: _single_cell_process_parallel(self, ii, jj, verbose)
            if not use_joblib:
                # Serial mode (fastest currently)
                for jj in jj_index:
                    if verbose:
                        print(ii, jj)
                    self.lat_mat_i[ii, jj] = self.lat_mat_i[jj,
                                                            ii] = lat_grid_compressed_i
                    self.lon_mat_i[ii, jj] = self.lon_mat_i[jj,
                                                            ii] = lon_grid_compressed_i
                    self.lat_mat_j[ii, jj] = self.lat_mat_j[jj,
                                                            ii] = self.lat_grid_compressed[jj]
                    self.lon_mat_j[ii, jj] = self.lon_mat_j[jj,
                                                            ii] = self.lon_grid_compressed[jj]
                    abs_x, x_j, x_i = scalar_cube_great_circle_distance(lat_grid_compressed_i,
                                                                        lon_grid_compressed_i,
                                                                        self.lat_grid_compressed[jj],
                                                                        self.lon_grid_compressed[jj],
                                                                        degree_dist=self.degree_dist,
                                                                        delta_x_method=self.delta_x_method,
                                                                        use_sklearn_haversine=self.use_sklearn_haversine)
                    if abs_x > self.max_dist:
                        self.cov_ns[ii, jj] = self.cov_ns[jj, ii] = 0.0
                    else:
                        sdev_j = self.sdev_local_estimates_compressed[jj]
                        sigma_parms_j = (self.Lx_local_estimates_compressed[jj],
                                         self.Ly_local_estimates_compressed[jj],
                                         self.theta_local_estimates_compressed[jj])
                        if verbose:
                            print(ii, jj, x_i, x_j)
                            print(ii, sdev_i, sigma_parms_i,
                                  np.rad2deg(sigma_parms_i[-1]))
                            print(jj, sdev_j, sigma_parms_j,
                                  np.rad2deg(sigma_parms_j[-1]))
                        # Compute eq 17 in Karspeck et al at each grid point
                        cov_bar = c_ij_anistropic_rotated_nonstationary(v,
                                                                        sdev_i,
                                                                        sdev_j,
                                                                        x_i,
                                                                        x_j,
                                                                        sigma_parms_i,
                                                                        sigma_parms_j,
                                                                        verbose=verbose)
                        ''' Fill in symmetric matrix '''
                        self.cov_ns[ii, jj] = self.cov_ns[jj, ii] = cov_bar
            else:
                if verbose:
                    for jj in jj_index:
                        print(ii, jj)

                def spnip(jj):
                    return self._single_cell_process_notinplace(ii, jj,
                                                                sdev_i,
                                                                sigma_parms_i,
                                                                lat_grid_compressed_i,
                                                                lon_grid_compressed_i,
                                                                verbose)
                parallel_kwargs = {'n_jobs': n_jobs, 'backend': backend}
                cov_bars = Parallel(
                    **parallel_kwargs)(delayed(spnip)(jj) for jj in jj_index)
                ''' Fill in symmetric matrix '''
                if verbose:
                    print(len(jj_index),
                          [cov_bars[jjjj] for jjjj in range(10)],
                          cov_bars[-1])  # For checking use
                for j_index, jj in enumerate(jj_index):
                    self.lat_mat_i[ii, jj] = self.lat_mat_i[jj,
                                                            ii] = self.lat_grid_compressed[ii]
                    self.lon_mat_i[ii, jj] = self.lon_mat_i[jj,
                                                            ii] = self.lon_grid_compressed[ii]
                    self.lat_mat_j[ii, jj] = self.lat_mat_j[jj,
                                                            ii] = self.lat_grid_compressed[jj]
                    self.lon_mat_j[ii, jj] = self.lon_mat_j[jj,
                                                            ii] = self.lon_grid_compressed[jj]
                    self.cov_ns[ii, jj] = cov_bars[j_index]
                    self.cov_ns[jj, ii] = cov_bars[j_index]

        cov_end_time = datetime.datetime.now()
        print('Cov processing ended: ',
              cov_end_time.strftime("%Y-%m-%d %H:%M:%S"))
        print('Time ellipsed: ', cov_end_time-cov_start_time)
        print('Mem used by cov mat = ', sizeof_fmt(sys.getsizeof(self.cov_ns)))

        # Code now reports eigvals and determinant of the constructed matrix
        self.cov_eig = np.sort(linalg.eigvalsh(self.cov_ns))
        self.cov_det = linalg.det(self.cov_ns)
        if self.check_positive_definite:
            # The purpose of this bit been replaced by repair_damaged_covariance.py
            # in production runs, but is still useful for unit tests
            # Perturb cov matrix to positive semi-definite if needed
            # Tests shows small negative eigval (most neg ~ -0.3 K**2) possible
            print('positive_definite_check is enabled')
            print('FYI, determinant = ', self.cov_det)
            print('FYI, eigenvalues sorted (first 10, last 10):')
            print(self.cov_eig[:10], '...', self.cov_eig[-10:])
            if np.min(self.cov_eig) < 0:
                # On the fly eigenvalue clipping
                print('Negative eigval detected; corrections will be applied.')
                self.positive_definite_check()
            else:
                print('Corrections are not needed.')
        else:
            print('positive_definite_check not enabled')
            print('FYI, determinant = ', self.cov_det)
            print('FYI, eigenvalues sorted (first 10, last 10):')
            print(self.cov_eig[:10], '...', self.cov_eig[-10:])
        print('Positive (semi-)definite checks complete.')

        # Compute correlation matrix
        print('Get reciprocal of covariance diagonal')
        sigma_inverse = np.diag(np.reciprocal(np.sqrt(np.diag(self.cov_ns))))
        print('Computing correlation matrix')
        self.cor_ns = sigma_inverse @ self.cov_ns @ sigma_inverse
        # Check for numerical errors
        print('Checking non-1 values in diagonal of correlation')
        diag_values = np.diag(self.cor_ns)
        where_not_one = diag_values != 1.0
        if np.any(where_not_one):
            largest_weird_value = np.max(np.abs(diag_values[where_not_one]))
            print('Ad hoc fix to numerical issues to corr matrix diag != 1.0')
            print('Largest error = ', largest_weird_value)
            np.fill_diagonal(self.cor_ns, 1.0)

    def _single_cell_process_notinplace(self,
                                        ii, jj,
                                        sdev_i,
                                        sigma_parms_i,
                                        lat_grid_compressed_i,
                                        lon_grid_compressed_i,
                                        verbose):
        ''' Standard safe way to do the covariance computation '''
        abs_x, x_j, x_i = scalar_cube_great_circle_distance(lat_grid_compressed_i,
                                                            lon_grid_compressed_i,
                                                            self.lat_grid_compressed[jj],
                                                            self.lon_grid_compressed[jj],
                                                            degree_dist=self.degree_dist,
                                                            delta_x_method=self.delta_x_method,
                                                            use_sklearn_haversine=self.use_sklearn_haversine)
        if abs_x > self.max_dist:
            cov_bar = 0.0
        else:
            sdev_j = self.sdev_local_estimates_compressed[jj]
            sigma_parms_j = (self.Lx_local_estimates_compressed[jj],
                             self.Ly_local_estimates_compressed[jj],
                             self.theta_local_estimates_compressed[jj])
            if verbose:
                print(ii, jj, x_i, x_j)
                print(ii, sdev_i, sigma_parms_i, np.rad2deg(sigma_parms_i[-1]))
                print(jj, sdev_j, sigma_parms_j, np.rad2deg(sigma_parms_j[-1]))
            # Compute eq 17 in Karspeck et al at each grid point
            cov_bar = c_ij_anistropic_rotated_nonstationary(self.v,
                                                            sdev_i, sdev_j,
                                                            x_i, x_j,
                                                            sigma_parms_i,
                                                            sigma_parms_j,
                                                            verbose=verbose)
        return cov_bar

    def _no_lazy_data(self, cube):
        ''' Disable iris cube lazy data '''
        if cube.has_lazy_data():
            cube.data  # pylint: disable=pointless-statement
        for coord in ['latitude', 'longitude']:
            if cube.coord(coord).has_lazy_points():
                cube.coord(coord).points  # pylint: disable=expression-not-assigned

    def _reverse_mask_from_compress_1D(self,
                                       compressed_1D_vector,
                                       fill_value=0.0,
                                       dtype=np.float32):
        '''
        A method called by remap_one_point_2_map
        It is used to create covariance/correlation maps from a single point

        DANGER WARNING, use different fill_value depending on situation
        This affects how signal and image processing module interacts with missing and masked values
        They don't ignore them, so a fill_value like 0 may be sensible for covariance (-999.99 will do funny things)
        iris doesn't care, but should use something like -999.99 or something
        '''
        compressed_counter = 0
        ans = np.zeros_like(self.cube_mask_1D, dtype=dtype)
        for i in range(len(self.cube_mask_1D)):
            if not self.cube_mask_1D[i]:
                ans[i] = compressed_1D_vector[compressed_counter]
                compressed_counter += 1
        ma.set_fill_value(ans, fill_value)
        ans = ma.masked_where(self.cube_mask_1D, ans)
        return ans

    def remap_one_point_2_map(self,
                              compressed_vector,
                              cube_name='stuff',
                              cube_unit='1'):
        '''
        This reverse one row/column of the covariance/correlation matrix
        to a plottable iris cube, using mask defined in class
        '''
        dummy_cube = self.Lx_local_estimates.copy()
        masked_vector = self._reverse_mask_from_compress_1D(compressed_vector)
        dummy_cube.data = masked_vector.reshape(self.xy_shape)
        dummy_cube.rename(cube_name)
        dummy_cube.units = cube_unit
        return dummy_cube

    def positive_definite_check(self):
        ''' 
        On the fly checking positive semidefinite and eigenvalue clipping 
        '''
        self.cov_ns = perturb_sym_matrix_2_positive_definite(self.cov_ns)
        self.cov_eig = np.sort(linalg.eigvalsh(self.cov_ns))
        self.cov_det = linalg.det(self.cov_ns)


def main():
    ''' MAIN '''
    print('=== Main ===')


if __name__ == "__main__":
    main()
