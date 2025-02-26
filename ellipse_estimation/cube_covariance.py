'''
Requires numpy, scipy, sklearm
iris needs to be installed (it is required by other modules within this package
xarray cubes should work via iris interface
'''
from collections import OrderedDict
import math
import warnings

import iris
import iris.coords as icoords
import iris.util as iutil
from cf_units import Unit
from joblib import Parallel, delayed
import numpy as np
from numpy import ma
from numpy import linalg
from scipy.special import kv as modified_bessel_2nd
from scipy.special import gamma
from scipy.spatial.transform import Rotation as R
from scipy import stats
from scipy.optimize import minimize
from sklearn.metrics.pairwise import haversine_distances, euclidean_distances
# from astropy.constants import R_earth

# Earth is nearly round.
# _RADIUS_OF_EARTH = R_earth.value # Radius along the equator 6378.1 km
_RADIUS_OF_EARTH = 6371000.0 # Average radius of Earth
_KM2M = 1000.0

## Each degree of latitude is equal to 60 nautical miles (with cosine correction for lon values)
nm_per_lat = 60.0 # 60 nautical miles per degree latitude
km2nm = 1.852 # 1852 meters per nautical miles

def _deg2nm(deg: float) -> float:
    '''
    deg: float (degrees)
    Convert degree latitude change to nautical miles
    '''
    return nm_per_lat * deg


def _deg2km(deg: float) -> float:
    '''
    deg: float (degrees)
    Convert degree latitude change to km
    '''
    return km2nm * _deg2nm(deg)


def _km2deg(km: float) -> float:
    '''
    km: float (km)
    Convert meridonal km change to degree latitude
    '''
    return (km / km2nm) / nm_per_lat


_default_n_jobs = 4
_default_backend = 'loky' ## loky appears to be fastest

model_type_2_supercategory = {'ps2006_kks2011_iso'  : '1_param_matern',
                              'ps2006_kks2011_ani'  : '2_param_matern',
                              'ps2006_kks2011_ani_r': '3_param_matern',
                              'ps2006_kks2011_iso_pd'  : '1_param_matern_pd',
                              'ps2006_kks2011_ani_pd'  : '2_param_matern_pd',
                              'ps2006_kks2011_ani_r_pd': '3_param_matern_pd',
                             }

fform_2_modeltype = {'anistropic_rotated': 'ps2006_kks2011_ani_r',
                     'anistropic'        : 'ps2006_kks2011_ani',
                     'isotropic'         : 'ps2006_kks2011_iso',
                     'anistropic_rotated_pd': 'ps2006_kks2011_ani_r_pd',
                     'anistropic_pd'        : 'ps2006_kks2011_ani_pd',
                     'isotropic_pd'         : 'ps2006_kks2011_iso_pd',
                    }

supercategory_parms = {'3_param_matern': OrderedDict([('Lx', Unit('degrees')),
                                                      ('Ly', Unit('degrees')),
                                                      ('theta', Unit('radians')),
                                                      ('standard_deviation', Unit('K')),
                                                      ('qc_code', Unit('1')),
                                                      ('number_of_iterations', Unit('1')),
                                                     ]),
                       '2_param_matern': OrderedDict([('Lx', Unit('degrees')),
                                                      ('Ly', Unit('degrees')),
                                                      ('standard_deviation', Unit('K')),
                                                      ('qc_code', Unit('1')),
                                                      ('number_of_iterations', Unit('1')),
                                                     ]),
                       '1_param_matern': OrderedDict([('rou', Unit('degrees')),
                                                      ('standard_deviation', Unit('K')),
                                                      ('qc_code', Unit('1')),
                                                      ('number_of_iterations', Unit('1')),
                                                     ]),
                       '3_param_matern_pd': OrderedDict([('Lx', Unit('km')),
                                                         ('Ly', Unit('km')),
                                                         ('theta', Unit('radians')),
                                                         ('standard_deviation', Unit('K')),
                                                         ('qc_code', Unit('1')),
                                                         ('number_of_iterations', Unit('1')),
                                                        ]),
                       '2_param_matern_pd': OrderedDict([('Lx', Unit('km')),
                                                         ('Ly', Unit('km')),
                                                         ('standard_deviation', Unit('K')),
                                                         ('qc_code', Unit('1')),
                                                         ('number_of_iterations', Unit('1')),
                                                        ]),
                       '1_param_matern_pd': OrderedDict([('rou', Unit('km')),
                                                         ('standard_deviation', Unit('K')),
                                                         ('qc_code', Unit('1')),
                                                         ('number_of_iterations', Unit('1')),
                                                        ]),
                      }
supercategory_nparms = {scp: len(supercategory_parms[scp]) for scp in supercategory_parms.keys()}

#def _AT_A(A):
#    return np.matmul(np.transpose(A), A)

#def _Dinv_A_Dinv(A):
#    sigma_inverse = np.zeros_like(A)
#    np.fill_diagonal(sigma_inverse, np.reciprocal(np.sqrt(np.diagonal(A))))
#    return np.matmul(np.matmul(sigma_inverse, A), sigma_inverse)

class CovarianceCube():
    def __init__(self, data_cube):

        ### Check input data_cube is actually usable
        self.tcoord_pos = -1
        self.xycoords_pos = []
        self.xycoords_name = []
        for i, coord in enumerate(data_cube.coords()):
            if coord.standard_name == 'time':
                self.tcoord_pos = i
            try:
                if ('lat' in coord.standard_name) or ('lon' in coord.standard_name):
                    self.xycoords_pos.append(i)
                    self.xycoords_name.append(coord.standard_name)
            except:
                pass
        if self.tcoord_pos == -1:
            raise ValueError('Input cube needs a time dimension')
        elif self.tcoord_pos != 0:
            raise ValueError('Input cube time dimension not at 0')
        if not self.xycoords_pos:
            raise ValueError('Input cube needs one spatial dimension')
        self.xycoords_pos = tuple(self.xycoords_pos)
        if 'lat' in self.xycoords_name[0]:
            self.xycoords_name = self.xycoords_name[::-1]

        ### Defining the input data
        self.data_cube = data_cube
        self.xy_shape = self.data_cube[0].shape
        assert len(self.xy_shape) == 2, "Time slices maps should be 2D; check if you have extra dimensions (ensemble/realizations?)"
        self.big_covar_size = np.prod(self.xy_shape)

        ### Detect data mask and determine dimension of array without masked data
        self.data_has_mask = ma.is_masked(self.data_cube.data) ## This is almost certain to be True anywhere near the coast
        if self.data_has_mask:
            ### Depending on dataset, the mask might not be invariant (like high latitude data with ice)
            ### xys with time varying mask are currently discarded. If analysis is conducted seasonally
            ### this should normally not a problem unless in high latitudes
            self.cube_mask = np.any(self.data_cube.data.mask, axis = 0)
            self.cube_mask_1D = self.cube_mask.flatten()
            self._self_mask()
            self.small_covar_size = np.sum(np.logical_not(self.cube_mask))
        else:
            self.cube_mask = np.zeros_like(data_cube[0].data.data, dtype = bool)
            self.cube_mask_1D = self.cube_mask.flatten()
            self.small_covar_size = self.big_covar_size

        ### Look-up table of the coordinates
        self.xx, self.yy = np.meshgrid(self.data_cube.coord('longitude').points,
                                       self.data_cube.coord('latitude').points)
        self.xm = ma.masked_where(self.cube_mask, self.xx)
        self.ym = ma.masked_where(self.cube_mask, self.yy)
        self.xy = np.column_stack([self.xm.compressed(), self.ym.compressed()])
        self.xy_full = np.column_stack([self.xm.flatten(), self.ym.flatten()])

        ## Length of time dimension
        self.time_n = len(self.data_cube.coord('time').points)

        ## Calculate the actual covariance and correlation matrix:
        self.Cov  = self._calc_cov()
        self.Corr = self._cov2cor()

    def _calc_cov(self, correlation=False, rounding=None):
        ## Reshape data to (t, xy), get rid of mask values -- cannot caculate cov for such data
        xyflatten_data = self.data_cube.data.reshape((len(self.data_cube.coord('time').points), self.big_covar_size))
        xyflatten_data = ma.compress_rowcols(xyflatten_data, -1)
        ## Remove mean -- even data that says "SST anomalies" don't have zero mean (?!)
        xy_mean = np.mean(xyflatten_data, axis = 0, keepdims = True)
        xyflatten_data = xyflatten_data - xy_mean
        xy_cov = np.matmul(np.transpose(xyflatten_data), xyflatten_data)
        #xy_cov = _AT_A(xyflatten_data)
        if correlation:
            return self._cov2cor(rounding=rounding)
        else:
            if rounding is not None:
                return np.round(xy_cov, rounding)
            else:
                return xy_cov

    def _cov2cor(self, rounding=None):
        '''
        https://gist.github.com/wiso/ce2a9919ded228838703c1c7c7dad13b
        '''
        sdevs = np.sqrt(np.diag(self.Cov))
        normalisation = np.outer(sdevs, sdevs)
        ans = self.Cov/normalisation
        ans[self.Cov==0] = 0
        # sigma_inverse = np.zeros_like(self.Cov)
        # np.fill_diagonal(sigma_inverse, np.reciprocal(np.sqrt(np.diagonal(self.Cov))))
        # ans = np.matmul(np.matmul(sigma_inverse, self.Cov), sigma_inverse)
        # #ans = _Dinv_A_Dinv(self.Cov)
        if rounding is not None:
            ans = np.round(ans, rounding)
        return ans

    def calc_distance_angle_selfcube(self, haversine=False, compressed=True):
        xx, yy = np.meshgrid(self.data_cube.coord('longitude').points,
                             self.data_cube.coord('latitude').points)
        xx = ma.masked_where(self.cube_mask, xx)
        yy = ma.masked_where(self.cube_mask, yy)
        if compressed:
            unmeshed_x = xx.compressed()
            unmeshed_y = yy.compressed()
        else:
            unmeshed_x = xx.flatten()
            unmeshed_y = yy.flatten()
        D, A = self._calc_distance_angle(unmeshed_x, unmeshed_y, haversine = haversine)
        return (D, A)

    def _calc_distance_angle(self, unmeshed_x, unmeshed_y, haversine = False):
        ## Compute a distance matrix between all points
        ## This is a memory demanding function!
        yx_og = np.column_stack([unmeshed_y, unmeshed_x])
        if haversine:
            ### Great circle - Earth
            def f_D(arr): return haversine_distances(arr) * _RADIUS_OF_EARTH/_KM2M
            yx = np.column_stack([np.array([np.deg2rad(lat) for lat in unmeshed_y]),
                                  np.array([np.deg2rad(lon) for lon in unmeshed_x])])
        else:
            ### Delta degrees - locally flat Earth, treating like it an image
            f_D = euclidean_distances
            yx = yx_og
        D = f_D(yx)
        ##
        ## Angle difference vs x/longitude-axis - below two yields the same result
        #_dy = euclidean_distances(np.column_stack([unmeshed_y, np.zeros_like(unmeshed_y)]))
        #_dx = euclidean_distances(np.column_stack([np.zeros_like(unmeshed_x), unmeshed_x]))
        _dy = euclidean_distances(np.column_stack([yx[:,0], np.zeros_like(unmeshed_y)]))
        _dx = euclidean_distances(np.column_stack([np.zeros_like(unmeshed_x), yx[:,1]]))
        dy = np.triu(_dy) - np.tril(_dy)
        dx = np.triu(_dx) - np.tril(_dx)
        A = np.arctan2(dy, dx) ### In Radians, note arctan2 (i.e. it can tell if dy and dx are negative)
        ##
        return (D, A) ### Both D and A are square matrices

    def _self_mask(self):
        broadcasted_mask = np.broadcast_to(self.cube_mask, self.data_cube.data.shape)
        self.data_cube.data = ma.masked_where(broadcasted_mask, self.data_cube.data)

    def _reverse_mask_from_compress_1D(self, compressed_1D_vector, fill_value = 0.0, dtype = np.float32):
        '''
        DANGER WARNING, use different fill_value depending on situation
        This affects how signal and image processing module interacts with missing and masked values
        They don't ignore them, so a fill_value like 0 may be sensible for covariance (-999.99 will do funny things)
        iris doesn't care, but should use something like -999.99 or something
        '''
        compressed_counter = 0
        ans = np.zeros_like(self.cube_mask_1D, dtype = dtype)
        for i in range(len(self.cube_mask_1D)):
            if not self.cube_mask_1D[i]:
                ans[i] = compressed_1D_vector[compressed_counter]
                compressed_counter += 1
        ma.set_fill_value(ans, fill_value)
        ans = ma.masked_where(self.cube_mask_1D, ans)
        return ans

    def remap_one_point_2_map(self, compressed_vector, cube_name = 'stuff', cube_unit = '1'):
        ''' 
        This reverse one row/column of the covariance/correlation matrix to a plottable iris cube 
        using mask defined in class
        '''
        dummy_cube = self.data_cube[0,:,:].copy()
        masked_vector = self._reverse_mask_from_compress_1D(compressed_vector)
        dummy_cube.data = masked_vector.reshape(self.xy_shape)
        dummy_cube.rename(cube_name)
        dummy_cube.units = cube_unit
        return dummy_cube

    def ps2006_kks2011_model(self,
                             xy_point,
                             max_distance=20.0,
                             min_distance=0.3,
                             v=3,
                             fform='anistropic_rotated',
                             unit_sigma=True,
                             delta_x_method="Modified_Met_Office",
                             guesses=None,
                             bounds=None,
                             opt_method='Nelder-Mead',
                             tol=0.001,
                             estimate_SE=None,
                             n_jobs=_default_n_jobs):
        #
        # Note on new tol kwarg:
        # For N-M, this sets the value to both xatol and fatol
        # Default is 1E-4 (?)
        # Since it affects accuracy of all values including rotation
        # rotation angle 0.001 rad ~ 0.05 deg
        #
        '''
        Fit ellipses/covariance models using adhoc local covariances

        the form of the covariance model depends on "fform"
        isotropic (radial distance only)
        anistropic (x and y are different, but not rotated)
        anistropic_rotated (rotated)

        add _pd to fform uses phyiscal distances instead of degrees
        without _pd, estimation uses degree lat lons

        range is defined max_distance (either in km and degrees)
        default is in degrees, but needs to be km if fform is from _pd series
        <--- likely to be wrong: max_distance should only be in degrees

        there is also a min_distance in which values, 
        matern function is not defined at the origin, so the 0.0 needs to removed

        v = matern covariance function shape parameter
        Karspeck et al and Paciorek and Schervish use 3 and 4
        but 0.5 and 1.5 are popular 
        0.5 gives an exponential decay
        lim v-->inf, Gaussian shape

        delta_x_method: only meaningful for _pd fits
        delta_x_method = "Spherical_COS_Law": uses COS(C) = COS(A)COS(B)
        delta_x_method = "Met_Office": Suspected flat (cylindrical) Earthers in Exeter! delta_x = 6400km x delta_lon (in radians)
        delta_x_method = "Modified_Met_Office": A bit better, but uses the average zonal dist at different lat
        '''
        ##
        matern = MLE_c_ij_Builder_Karspeck(v, fform, unit_sigma=unit_sigma)
        lonlat = self.xy[xy_point]
        correlation_vector = self.Corr[xy_point, :]
        ##
        R2 = self.data_cube[0].copy()
        R2x = self._reverse_mask_from_compress_1D(self.Corr[xy_point, :])
        R2.data = R2x.reshape(self.xy_shape)
        R2.units = "1"
        ##
        ## dx and dy are in degrees
        dx = np.array([a - lonlat[0] for a in self.xy[:, 0]])
        dy = np.array([a - lonlat[1] for a in self.xy[:, 1]])
        dx[dx >  180.0] -= 360.0
        dx[dx < -180.0] += 360.0
        ## Delete the origin (can't have dx = dy = 0)
        dx = np.delete(dx, xy_point)
        dy = np.delete(dy, xy_point)
        correlation_vector2 = np.delete(correlation_vector, xy_point)
        ##
        ## distance is in delta-degrees
        lonlat_vector = np.column_stack([dx, dy])
        distance = linalg.norm(lonlat_vector, axis=1)
        #distance_i = np.abs(dx)
        #distance_j = np.abs(dy)
        distance_i = dx
        distance_j = dy
        dx_sign = np.sign(dx)
        ##
        if fform[-3:] == '_pd':
            ''' 
            There are two ways to do that 
            (1) Use the Haversine formula and solve for distance_ii 
            (2) Use the law of cosines, taking inputs of distance_ii and distance_jj and solve for radial distance <-- this retains the up and down.... 
            '''
            ''' 
            Haversine -- (delta_lat, delta_lon)
            sklearn.metrics.pairwise.haversine_distances(X, Y=None)[source]
            Compute the Haversine distance between samples in X and Y.
            The Haversine (or great circle) distance is the angular distance between two points on the surface of a sphere. 
            The first coordinate of each point is assumed to be the latitude, the second is the longitude, given in radians. 
            The dimension of the data must be 2.
            '''
            latlon_vector2 = np.column_stack([np.deg2rad(lonlat[1]+dy),
                                              np.deg2rad(lonlat[0]+dx)])
            latlon2 = np.array([np.deg2rad(lonlat[1]),
                                np.deg2rad(lonlat[0])])
            latlon2 = latlon2[np.newaxis, :]
            X_train_radial = haversine_distances(latlon_vector2, latlon2)[:, 0]
            distance_jj = np.deg2rad(distance_j)
            ''' 
            Law of cosines/Pyth Theroem on a sphere surface:
            https://sites.math.washington.edu/~king/coursedir/m445w03/class/02-03-lawcos-answers.html
            COS(Haversine Dist) = COS(Delta_Lat) COS(Delta_X)
            
            Note: 
            Delta_X != Delta_LON or Delta_Lon x COS LAT
            Delta_X itself is a great circle distance
            Here meridonal displacement is always defined relative to the north and south pole!
            '''
            if delta_x_method == "Spherical_COS_Law":
                ## This doesn't appears to work... recheck needed
                inside_arccos = np.cos(X_train_radial)/np.cos(distance_jj)
                print('Check, num of inside_arccos vals = ',len(inside_arccos))
                print('Check, num of abs(inside_arccos) > 1.0 = ',np.sum(np.abs(inside_arccos) > 1.0))
                print('Check, max(inside_arccos): max(inside_arccos) = ',inside_arccos.max())
                ## Numerical issues may lead to numbers slightly greater than 1.0 or less than -1.0
                inside_arccos[inside_arccos >  1.0] =  1.0
                inside_arccos[inside_arccos < -1.0] = -1.0
                distance_ii = dx_sign * np.arccos(inside_arccos)
            elif delta_x_method == "Met_Office":
                ## Cylindrical approximation
                distance_ii = np.deg2rad(dx)
            elif delta_x_method == "Modified_Met_Office":
                average_cos = 0.5 * (np.cos(np.deg2rad(lonlat[1]+dy)) + np.cos(np.deg2rad(lonlat[1])))
                distance_ii = np.deg2rad(dx) * average_cos
            else:
                raise ValueError('Unknown delta_x_method')
            ##
            ## Converts dx and dy to physical distance (km)
            X_train_directional = np.column_stack([distance_ii*_RADIUS_OF_EARTH/_KM2M,
                                                   distance_jj*_RADIUS_OF_EARTH/_KM2M])
            X_train_radial = X_train_radial*_RADIUS_OF_EARTH/_KM2M
        else:
            X_train_directional = np.column_stack([distance_i, distance_j])
            X_train_radial = distance.copy()
        y_train = correlation_vector2.copy()
        ##
        print('Calculation check for X_train_directional')
        print(X_train_directional.shape)
        print('i-th component range, min, max: ',
              np.ptp(X_train_directional[:,0]),
              np.min(X_train_directional[:,0]),
              np.max(X_train_directional[:,0]))
        print('j-th component range, min, max: ',
              np.ptp(X_train_directional[:,1]),
              np.min(X_train_directional[:,1]),
              np.max(X_train_directional[:,1]))
        distance_limit = np.where(distance > max_distance)[0].tolist()
        distance_threshold = np.where(distance < min_distance)[0].tolist()
        xys_2_drop = list(set(distance_limit+distance_threshold))
        X_train_directional = np.delete(X_train_directional, xys_2_drop, axis=0)
        X_train_radial = np.delete(X_train_radial, xys_2_drop, axis=0)
        y_train = np.delete(y_train, xys_2_drop)
        ##
        model_type = fform_2_modeltype[fform]
        if (fform == 'anistropic_rotated') or (fform == 'anistropic_rotated_pd'):
            X_train = X_train_directional
        elif (fform == 'anistropic') or (fform == 'anistropic_pd'):
            X_train = X_train_directional
        elif (fform == 'isotropic') or (fform == 'isotropic_pd'):
            X_train = X_train_radial.reshape(-1, 1)
        else:
            raise ValueError('Unknown fform')
        ##
        results, _, bbs = matern.fit(X_train,
                                     y_train,
                                     guesses=guesses,
                                     bounds=bounds,
                                     opt_method=opt_method,
                                     tol=tol,
                                     estimate_SE=estimate_SE,
                                     n_jobs=n_jobs)
        ##
        if unit_sigma:
            model_params = results.x.tolist()
            std = None
        else:
            model_params = results.x.tolist()[:-1]
            std = results.x.tolist()[-1]
        '''
        0: success
        1: success but with one parameter reaching lower bounadries
        2: success but with one parameter reaching upper bounadries
        3: success with multiple parameters reaching the boundaries (aka both Lx and Ly), can be both at lower or upper boundaries
        9: fail, probably due to running out of maxiter (see scipy.optimize.minimize kwargs "options)"
        '''
        if results.success:
            fit_success2 = 0.0
            for model_param, bb in zip(model_params, bbs):
                left_check = math.isclose(model_param, bb[0], rel_tol=0.01)
                right_check = math.isclose(model_param, bb[1], rel_tol=0.01)
                left_advisory = 'near_left_bnd' if left_check else 'not_near_left_bnd'
                right_advisory = 'near_right_bnd' if right_check else 'not_near_rgt_bnd'
                print('Convergence success after ', results.nit, ' iterations :) : ',
                      model_param, bb[0], bb[1],
                      left_advisory, right_advisory)
                if left_check:
                    if fit_success2 == 0.0:
                        fit_success2 = 1.0
                    else:
                        fit_success2 = 3.0
                elif right_check:
                    if fit_success2 == 0.0:
                        fit_success2 = 2.0
                    else:
                        fit_success2 = 3.0
                else:
                    pass
            print('RMSE of multivariate norm fit = ', std)
        else:
            print('Convergence fail after ', results.nit, ' iterations :(.')
            print(model_params)
            fit_success2 = 9
        print('QC flag = ', fit_success2)
        model_params.append(np.sqrt(self.Cov[xy_point, xy_point]/self.time_n))  # append standard deviation
        model_params.append(fit_success2)
        model_params.append(results.nit)
        ##
        v_coord = make_v_aux_coord(v)
        template_cube = self._make_template_cube(xy_point)
        model_as_cubelist = create_output_cubes(template_cube,
                                                model_type=model_type,
                                                additional_meta_aux_coords=[v_coord],
                                                default_values=model_params)['param_cubelist']
        ##
        return {'Correlation': R2,
                'MaternObj': matern,
                'Model': results,
                'Model_Type': model_type, 
                'Model_as_1D_cube': model_as_cubelist}

    def find_nearest_xy_index_in_cov_matrix(self, lonlat, use_full = False):
        ## lonlat = [lon, lat]
        if use_full:
            a = self.xy_full
        else:
            a = self.xy
        lonlat = np.asarray(lonlat)
        idx = ((a[:,0] - lonlat[0])**2.0 + (a[:,1] - lonlat[1])**2.0).argmin()
        return (idx, a[idx,:])

    def _xy_2_xy_full_index(self, xy_point):
        ans = int(np.argwhere(np.all((self.xy_full - self.xy[xy_point,:]) == 0, axis=1))[0])
        return ans

    def _make_template_cube(self, xy_point):
        xy = self.xy[xy_point, :]
        return self._make_template_cube2(xy)
        #t_len = len(self.data_cube.coord('time').points)
        #template_cube = self.data_cube[t_len//2].intersection(longitude = (xy[0]-0.05, xy[0]+0.05), latitude = (xy[1]-0.05, xy[1]+0.05))
        #return template_cube

    def _make_template_cube2(self, lonlat):
        xy = lonlat
        t_len = len(self.data_cube.coord('time').points)
        template_cube = self.data_cube[t_len//2].intersection(longitude=(xy[0]-0.05, xy[0]+0.05),
                                                              latitude=(xy[1]-0.05, xy[1]+0.05))
        return template_cube

    def __str__(self):
        return str(self.__class__)
        #return str(self.__class__) + ": " + str(self.__dict__)


def sigma_rot_func(Lx, Ly, theta):
    '''
    Equation 15 in Karspeck el al 2011 and Equation 6
    in Paciorek and Schervish 2006,
    assuming Sigma(Lx, Ly, theta) locally/moving-window invariant or
    we have already taken the mean (Sigma overbar, PP06 3.1.1)
    '''
    # sigma is a 2x2 matrix
    if theta is None:
        theta = 0.0
    r = R.from_rotvec([0, 0, theta])
    r_matrix_2d = r.as_matrix()[:2, :2]
    sigma = np.matmul(np.matmul(r_matrix_2d,
                                np.diag([Lx**2, Ly**2])),
                      np.transpose(r_matrix_2d))
    return sigma


def mahal_dist_func_rot(delta_x, delta_y, Lx, Ly, theta=None, verbose=False):
    '''
    Calculate tau if Lx, Ly, theta is known (aka, this takes the additional step to compute sigma)
    This is needed for MLE estimation of Lx, Ly, and theta

    for MLE:
    d is distance between two points
    Lx, Ly, theta are unknown parameters that need to be estimated
    and replaces d/rou in equation 14 in Karspect et al paper
    '''
    # sigma is 4x4 matrix
    if theta is not None:
        sigma = sigma_rot_func(Lx, Ly, theta)
    else:
        sigma = np.diag(np.array([Lx**2.0, Ly**2.0]))
    xi_minus_xj = np.array([delta_x, delta_y])
    tau_inside_squareroot = np.matmul(np.matmul(np.transpose(xi_minus_xj),
                                                linalg.inv(sigma)),
                                      xi_minus_xj)
    tau = np.sqrt(tau_inside_squareroot)
    if verbose:
        print('tau', tau)
    return tau  # tau is a scalar


def mahal_dist_func_sigma(delta_x, delta_y, sigma, verbose=False):
    '''
    Calculate tau directly if sigma is already known
    Useful for sigma_bar computations
    '''
    xi_minus_xj = np.array([delta_x, delta_y])
    tau_inside_squareroot = np.matmul(np.matmul(np.transpose(xi_minus_xj), linalg.inv(sigma)), xi_minus_xj)
    tau = np.sqrt(tau_inside_squareroot)
    if verbose:
        print('tau', tau)
    return tau


def mahal_dist_func(delta_x, delta_y, Lx, Ly):
    return mahal_dist_func_rot(delta_x, delta_y, Lx, Ly, theta=None)


def make_v_aux_coord(v):
    return icoords.AuxCoord(v, long_name='v_shape', units='no_unit')


def c_ij_anistropic_rotated(v,
                            sdev,
                            x_i, x_j,
                            Lx, Ly,
                            theta,
                            sdev_j=None,
                            verbose=False):
    '''
    Covariance structure between base point i and j
    Assuming local stationarity or slowly varing
    so that some terms in PS06 drops off (like Sigma_i ~ Sigma_j instead of treating them as different)
    (aka second_term below)
    this makes formulation a lot simplier
    We let sdev_j opens to changes, 
    but in pracitice, we normalise everything to correlation so sdev == sdev_j == 1
    '''
    ##
    if sdev_j is None:
        sdev_j = sdev
    ##
    #sigma = sigma_rot_func(Lx, Ly, theta)
    tau = mahal_dist_func_rot(x_i, x_j, Lx, Ly, theta=theta, verbose=verbose)
    ##
    first_term = (sdev * sdev_j)/(gamma(v)*(2.0**(v-1)))
    '''
    If data is assumed near stationary locally, sigma_i ~ sigma_j same
    making (sigma_i)**1/4 (sigma_j)**1/4 / (mean_sigma**1/2) = 1.0
    Treating it the otherwise is a major escalation to the computation
    See discussion 2nd paragraph in 3.1.1 in Paciroke and Schervish 2006
    '''
    #second_term = 1.0
    third_term = (2.0*tau*np.sqrt(v))**v
    forth_term = modified_bessel_2nd(v, 2.0*tau*np.sqrt(v))
    ans = first_term * third_term * forth_term
    #ans = first_term * second_term * third_term * forth_term
    ##
    if verbose:
        print('first_term', first_term, first_term.shape)
        print('third_term', third_term, third_term.shape)
        print('forth_term', forth_term, forth_term.shape)
        print('ans', ans, ans.shape)
    return ans

def c_ij_anistropic_unrotated(v, sdev, x_i, x_j, Lx, Ly, sdev_j = None):
    return c_ij_anistropic_rotated(v, sdev, x_i, x_j, Lx, Ly, None, sdev_j=sdev_j)

def c_ij_istropic(v, sdev, displacement, rou, sdev_j=None):
    ##
    if sdev_j is None:
        sdev_j = sdev
    ##
    tau = np.abs(displacement)/rou
    ##
    first_term = (sdev * sdev_j)/(gamma(v)*(2.0**(v-1)))
    third_term = (2.0*tau*np.sqrt(v))**v
    forth_term = modified_bessel_2nd(v, 2.0*tau*np.sqrt(v))
    ans = first_term * third_term * forth_term
    return ans

class MLE_c_ij_Builder_Karspeck():
    ''' This class assumes your input to be a standardised correlation matrix '''
    def __init__(self, v, fform, standardised_cov_matrix=True, unit_sigma=True):
        self.fform = fform
        self.unit_sigma = unit_sigma
        if fform == 'isotropic':
            self.n_params = 1
            self.default_guesses = [7.0]
            self.default_bounds  = [(0.5, 50),]
            if standardised_cov_matrix:
                self.c_ij = lambda X, rou: c_ij_istropic(v, 1, X, rou)
            else:
                raise NotImplementedError('Standardised/normalise covariance matrix first to correlation matrix')
        elif fform == 'isotropic_pd':
            self.n_params = 1
            self.default_guesses = [_deg2km(7.0)]
            self.default_bounds  = [(_deg2km(0.5), _deg2km(50)),]
            if standardised_cov_matrix:
                self.c_ij = lambda X, rou: c_ij_istropic(v, 1, X, rou)
            else:
                raise NotImplementedError('Standardised/normalise covariance matrix first to correlation matrix')
        elif fform == 'anistropic':
            ## anistropic non-rotated
            self.n_params = 2
            self.default_guesses = [7.0, 7.0]
            self.default_bounds  = [(0.5, 50), (0.5, 30)]
            if standardised_cov_matrix:
                self.c_ij = lambda X, Lx, Ly: c_ij_anistropic_unrotated(v, 1, X[0], X[1], Lx, Ly)
            else:
                raise NotImplementedError('Standardised/normalise covariance matrix first to correlation matrix')
        elif fform == 'anistropic_pd':
            ## anistropic non-rotated
            self.n_params = 2
            self.default_guesses = [_deg2km(7.0), _deg2km(7.0)]
            self.default_bounds  = [(_deg2km(0.5), _deg2km(50)), (_deg2km(0.5), _deg2km(30))]
            if standardised_cov_matrix:
                self.c_ij = lambda X, Lx, Ly: c_ij_anistropic_unrotated(v, 1, X[0], X[1], Lx, Ly)
            else:
                raise NotImplementedError('Standardised/normalise covariance matrix first to correlation matrix')
        elif fform == 'anistropic_rotated':
            ## anistropic rotated
            self.n_params = 3
            self.default_guesses = [7.0, 7.0, 0.0]
            self.default_bounds  = [(0.5, 50), (0.5, 30), (-2.0*np.pi, 2.0*np.pi)]
            if standardised_cov_matrix:
                self.c_ij = lambda X, Lx, Ly, theta: c_ij_anistropic_rotated(v, 1, X[0], X[1], Lx, Ly, theta)
            else:
                raise NotImplementedError('Standardised/normalise covariance matrix first to correlation matrix')
        elif fform == 'anistropic_rotated_pd':
            ## anistropic rotated
            self.n_params = 3
            self.default_guesses = [_deg2km(7.0), _deg2km(7.0), 0.0]
            self.default_bounds  = [(_deg2km(0.5), _deg2km(50)), (_deg2km(0.5), _deg2km(30)), (-2.0*np.pi, 2.0*np.pi)]
            if standardised_cov_matrix:
                self.c_ij = lambda X, Lx, Ly, theta: c_ij_anistropic_rotated(v, 1, X[0], X[1], Lx, Ly, theta)
            else:
                raise NotImplementedError('Standardised/normalise covariance matrix first to correlation matrix')
        else:
            raise ValueError('Unknown fform')

    def negativeloglikelihood(self, X, y,
                              params,
                              arctanh_transform=True,
                              backend=_default_backend,
                              n_jobs=_default_n_jobs):
    ###
    ### log(LL) = SUM (f (y,x|params) )
    ### params = Maximise (log(LL))
    ### params = Minimise (-log(LL)) <--- which is how usually the computer solves it
    ### assuming errors of params are normally distributed
    ###
    ### There is a hidden scale/standard deviation in stats.norm.logpdf (scale, which defaults to 1)
    ### but since we have scaled our values to covariance to correlation (and even used
    ### Fisher transform) as part of the function, it can be dropped
    ###
    ### Otherwise, you need to have sdev as the last value of params, and should be set to the
    ### scale parameter
    ###
        if self.n_params == 1:
            rou = params[0]
            if self.unit_sigma:
                sigma = 1
            else:
                sigma = params[1]
            y_LL = self.c_ij(X, rou)
        elif self.n_params == 2:
            Lx = params[0]
            Ly = params[1]
            if self.unit_sigma:
                sigma = 1
            else:
                sigma = params[2]
            y_LL = Parallel(n_jobs=n_jobs, backend=backend)(delayed(self.c_ij)(X[n_x_j, :], Lx, Ly) for n_x_j in range(X.shape[0]))
            #y_LL = []
            #for n_x_j in range(X.shape[0]):
            #    y_LL.append(self.c_ij(X[n_x_j, :], Lx, Ly))
            y_LL = np.array(y_LL)
        elif self.n_params == 3:
            Lx = params[0]
            Ly = params[1]
            theta = params[2]
            if self.unit_sigma:
                sigma = 1
            else:
                sigma = params[3]
            y_LL = Parallel(n_jobs=n_jobs, backend=backend)(delayed(self.c_ij)(X[n_x_j, :], Lx, Ly, theta) for n_x_j in range(X.shape[0]))
            #y_LL = []
            #for n_x_j in range(X.shape[0]):
            #    y_LL.append(self.c_ij(X[n_x_j,:], Lx, Ly, theta))
            y_LL = np.array(y_LL)
        '''
        if y is correlation,
        it might be useful to Fisher transform them before plugging into norm.logpdf
        this affects values close to 1 and -1
        imposing better behavior to the differences at the tail
        '''
        if arctanh_transform:
            #
            # Warning against arctanh(abs(y) > 1); (TO DO: Add correction later)
            arctanh_threshold = 0.999999
            # arctanh_threshold = 1.0
            max_abs_y = np.max(np.abs(y))
            max_abs_yLL = np.max(np.abs(y_LL))
            if max_abs_y >= arctanh_threshold:
                warn_msg = 'abs(y) >= '+str(arctanh_threshold)+' detected; '
                warn_msg += 'fudged to threshold; max(abs(y))='+str(max_abs_y)
                warnings.warn(warn_msg, RuntimeWarning)
                y[np.abs(y)>arctanh_threshold] = np.sign(y[np.abs(y)>arctanh_threshold]) * arctanh_threshold
                # y[np.abs(y) > 1] = np.sign(y[np.abs(y) > 1]) * 0.9999
            # if np.any(np.isclose(np.abs(y), 1.0)):
            #     warn_msg = 'abs(y) is close to 1; max(abs(y))='+str(max_abs_y)
            #     warnings.warn(warn_msg, RuntimeWarning)
            #     y[np.isclose(np.abs(y), 1.0)] = np.sign(y[np.isclose(np.abs(y), 1.0)]) * 0.9999
            #
            if max_abs_yLL >= 1:
                warn_msg = 'abs(y_LL) >= '+str(arctanh_threshold)+' detected; '
                warn_msg += 'fudged to threshold; max(abs(y_LL))='+str(max_abs_yLL)
                warnings.warn(warn_msg, RuntimeWarning)
                y_LL[np.abs(y_LL)>arctanh_threshold] = np.sign(y_LL[np.abs(y_LL)>arctanh_threshold]) * arctanh_threshold
                # y_LL[np.abs(y_LL) > 1] = np.sign(y_LL[np.abs(y_LL) > 1]) * 0.9999
            # if np.any(np.isclose(np.abs(y_LL), 1.0)):
            #     warn_msg = 'abs(y_LL) close to 1 detected; max(abs(y_LL))='+str(max_abs_yLL)
            #     warnings.warn(warn_msg, RuntimeWarning)
            #     y_LL[np.isclose(np.abs(y_LL), 1.0)] = np.sign(y_LL[np.isclose(np.abs(y_LL), 1.0)]) * 0.9999
            #
            nLL = -1.0 * np.sum(stats.norm.logpdf(np.arctanh(y), loc=np.arctanh(y_LL), scale=sigma))
        else:
            nLL = -1.0 * np.sum(stats.norm.logpdf(y, loc=y_LL, scale=sigma))
        return nLL

    def build_negativeloglikelihood_for_optimisation(self, X, y,
                                                     n_jobs=_default_n_jobs,
                                                     backend=_default_backend):
        return lambda params: self.negativeloglikelihood(X, y,
                                                         params,
                                                         n_jobs=n_jobs,
                                                         backend=backend)

    def fit(self,
            X, y,
            guesses=None,
            bounds=None,
            opt_method='Nelder-Mead',
            tol=None,
            estimate_SE='bootstrap_parallel',
            n_sim=500,
            n_jobs=_default_n_jobs,
            backend=_default_backend,
            random_seed=1234):
        ##
        '''
        Default solver in Nelder-Mead as used in the Karspeck paper
        https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html
        default max-iter is 200 x (number_of_variables)
        for 3 variables (Lx, Ly, theta) --> 200x3 = 600
        note: unlike variogram fitting, no nugget, no sill, and no residue variance (normalised data but Fisher transform needed?)
        can be adjusted using "maxiter" within "options" kwargs
        '''
        ##
        if guesses is None:
            guesses = self.default_guesses
        else:
            guesses = list(guesses)
        if bounds is None:
            bounds = self.default_bounds
        else:
            bounds = list(bounds)
        if not self.unit_sigma:
            guesses.append(0.1)
            bounds.append((0.0001, 0.5))
        ##
        LL_observedXy_unknownparams = self.build_negativeloglikelihood_for_optimisation(X, y,
                                                                                        n_jobs=n_jobs,
                                                                                        backend=backend)
        ##
        print('X range: ', np.min(X), np.max(X))
        print('y range: ', np.min(y), np.max(y))
        zipper = zip(guesses, bounds)
        for g, b in zipper:
            print('init: ', g, 'bounds: ', b)
        ##
        results = minimize(LL_observedXy_unknownparams,
                           guesses,
                           bounds=bounds,
                           method=opt_method,
                           tol=tol)
        ##
        ''' This does not account for standard errors in the correlation/covariance matrix ! '''
        if estimate_SE is not None:
            if estimate_SE == 'bootstrap_serial':
                ### Serial
                sim_params = []
                for looper in range(n_sim):
                    sim_params.append(self._bootstrap_once(X, y, guesses, bounds, opt_method, tol=tol, seed=random_seed+looper))
                sim_params = np.array(sim_params)
                SE = np.std(sim_params, axis = 0)
            elif estimate_SE == 'bootstrap_parallel':
                ### Parallel
                ### On JASMIN Jupyter: n_jobs = 5 leads to 1/3 wallclock time
                kwargs_0 = {'n_jobs': n_jobs, 'backend': backend}
                workers = range(n_sim)
                sim_params = Parallel(**kwargs_0)(delayed(self._bootstrap_once)(X, y,
                                                                                guesses,
                                                                                bounds,
                                                                                opt_method,
                                                                                tol=tol,
                                                                                seed=random_seed+worker) for worker in workers)
                sim_params = np.array(sim_params)
                SE = np.std(sim_params, axis=0)
            elif estimate_SE == 'hessian':
                ''' note: autograd does not work with scipy's Bessel functions '''
                raise NotImplementedError('Second order deriviative (Hessian) of Fisher Information not implemented')
            else:
                raise ValueError('Unknown estimate_SE')
            return [results, SE, bounds]
        else:
            print('Standard error estimates not required')
            return [results, None, bounds]

    def _bootstrap_once(self, X, y, guesses, bounds, opt_method, tol=None, seed=1234):
        rng = np.random.RandomState(seed) # pylint: disable=no-member
        len_obs = len(y)
        i_obs = np.arange(len_obs)
        bootstrap_i = rng.choice(i_obs, size=len_obs, replace=True)
        X_boot = X[bootstrap_i, ...]
        y_boot = y[bootstrap_i]
        LL_bootXy_simulatedparams = self.build_negativeloglikelihood_for_optimisation(X_boot, y_boot)
        ans = minimize(LL_bootXy_simulatedparams, guesses, bounds=bounds, method=opt_method, tol=tol)
        return ans.x

'''
For data presentation 
'''
def create_output_cubes(template_cube,
                        model_type='ps2006_kks2011_iso',
                        default_values=[-999.9, -999.9, -999.9],
                        additional_meta_aux_coords=None,
                        dtype=np.float32):
    ##
    supercategory = model_type_2_supercategory[model_type]
    params_dict = supercategory_parms[supercategory]
    ##
    model_type_coord = icoords.AuxCoord(model_type,
                                        long_name='fitting_model',
                                        units='no_unit')
    supercategory_coord = icoords.AuxCoord(supercategory,
                                           long_name='supercategory_of_fitting_model',
                                           units='no_unit')
    ##
    ans_cubelist = iris.cube.CubeList()
    ans_paramlist = []
    ##
    for param, default_value in zip(params_dict.keys(), default_values):
        ans_paramlist.append(param)
        param_cube = template_cube.copy()
        param_cube.rename(param)
        param_cube.long_name = param
        param_cube.add_aux_coord(model_type_coord)
        param_cube.add_aux_coord(supercategory_coord)
        param_cube.units = params_dict[param]
        if additional_meta_aux_coords is not None:
            for add_coord in additional_meta_aux_coords:
                param_cube.add_aux_coord(add_coord)
        if param_cube.ndim == 0:
            param_cube = iutil.new_axis(iutil.new_axis(param_cube, 'longitude'), 'latitude')
        param_cube.data[:] = default_value
        param_cube.data = param_cube.data.astype(dtype)
        if ma.isMaskedArray(template_cube.data):
            param_cube.data.mask = template_cube.data.mask
        ans_cubelist.append(param_cube)
    return {'param_cubelist': ans_cubelist, 'param_names': ans_paramlist}


def main():
    print('=== Main ===')
    #_test_load()


if __name__ == "__main__":
    main()
