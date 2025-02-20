import numbers

import iris
import numpy as np
from numpy.random import multivariate_normal
from scipy import stats
import xarray as xa

import cube_covariance
import cube_covariance_nonstationary_stich as ccns

def fill_cube_with_uniform_parms(cube_template,
                                 sdev,
                                 sigma_parms):
    '''
    Return 4 cubes (sdev, Lx, Ly, theta) for each grid box of the template cube

    :param cube_template: instance of iris.Cube, the template domain to be filled
    :param sdev: float, standard deviation
    :param sigma_parms: iterable of length 3, (Lx, Lx, theta)
    :returns: a single instance of iris.Cube.CubeList with filled sdev, Lx, Ly, theta
    '''
    #
    if isinstance(cube_template, xa.DataArray):
        cube_template = cube_template.to_iris()
    assert isinstance(cube_template, iris.cube.Cube), 'cube_template should be an instance of xarray.DataArray or iris.cube'
    #
    assert len(sigma_parms) == 3, 'Length of sigma_parms must be 3'
    for sparm_i, sparm in enumerate([sdev, sigma_parms[0], sigma_parms[1], sigma_parms[2]]):
        assert isinstance(sparm, numbers.Number), 'Non number detected on idx '+str(sparm_i)
    #
    print('Creating parameter cubes with uniform parameters')
    print('sdev = ', sdev)
    print('(Lx, Ly, theta) = ', sigma_parms)
    #
    ans = iris.cube.CubeList()
    #
    model_parms = [sdev, sigma_parms[0], sigma_parms[1], sigma_parms[2]]
    cube_names = ['standard_deviation', 'Lx', 'Ly', 'theta']
    cube_units = ['K', 'km', 'km', 'radians']
    zipper = zip(model_parms, cube_names, cube_units)
    for sparm, cname, cunit in zipper:
        ebuc = cube_template.copy()
        ebuc.rename(cname)
        ebuc.units = cunit
        ebuc.data[:] = sparm
        ans.append(ebuc)
    #
    print(ans)
    return ans

'''
A covariance based on outputs generated from fill_cube_with_uniform_parms 
can be generated using 
cube_covariance_nonstationary_stich CovarianceCube_PreStichedLocalEstimates Class

CovarianceCube_PreStichedLocalEstimates calls a distance function within compute_distance 
'''

class _EllipseSimulation():

    def __init__(self, v, sdev_cube, Lx_cube, Ly_cube, theta_cube):
        assert isinstance(v, numbers.Number), 'Non number detected for v'
        assert v > 0.0, 'v must be a positive number (0 == bad)'
        self.v = v
        self.sdev = sdev_cube
        self.Lx = Lx_cube
        self.Ly = Ly_cube
        self.theta = theta_cube
        print(self.v)
        print(repr(self.sdev))
        print(repr(self.Lx))
        print(repr(self.Ly ))
        print(repr(self.theta))

    def create_cov(self, kwargs4CC_PLE=None):
        print('Creating stiched covariance and correlation using CovarianceCube_PreStichedLocalEstimates.')
        if kwargs4CC_PLE is None:
            kwargs4CC_PLE = {'delta_x_method': 'Modified_Met_Office',
                             'check_positive_definite': True}
        self.CCPLE_out = ccns.CovarianceCube_PreStichedLocalEstimates(self.Lx,
                                                                      self.Ly,
                                                                      self.theta,
                                                                      self.sdev,
                                                                      v=self.v,
                                                                      **kwargs4CC_PLE)


    def simulate_cov(self,
                     mean=0.0,
                     size=1,
                     reshaped_2_og=True):
        if not self.check_self_exist(self.CCPLE_out):
            print('Covariance and correlation not created yet; creating with default kwargs.')
            self.create_cov()
        try:
            iter(mean)
        except TypeError:
            mean = np.array([mean]*self.CCPLE_out.cov_ns.shape[0])
        ans = multivariate_normal(mean, self.CCPLE_out.cov_ns, size=size)
        print(ans.shape)
        if reshaped_2_og:
            ans = ans.reshape((size, self.Lx.shape[0], self.Lx.shape[1]))
            print(ans.shape)
            return ans
        return ans


    def simulated_map_as_iris_cube_with_fake_t(self, map_arr, t_coord):
        if map_arr.shape[-2:] != self.Lx.shape:
            print('map_arr.shape[-2:] = ', map_arr.shape[-2:])
            print('Lx.shape = ', self.Lx.shape)
            raise ValueError('map_arr shape has unexpected shape')
        y_coord = self.Lx.coord('latitude')
        x_coord = self.Lx.coord('longitude')
        ans = iris.cube.Cube(map_arr, dim_coords_and_dims=[(t_coord, 0), (y_coord, 1), (x_coord, 2)])
        ans.rename('Simulated values')
        ans.units = '1'
        return ans


    def check_self_exist(self, thing):
        try:
            thing
        except NameError:
            return False
        return True


    def __repr__(self):
        return f'SuperClass _EllipseSimulation with v = {self.v}'


class EllipseSimulation_UniformParms(_EllipseSimulation):

    def __init__(self,
                 v,
                 cube_template,
                 sdev_uniform,
                 sigma_parms_uniform):
        parm_cubes = fill_cube_with_uniform_parms(cube_template,
                                                  sdev_uniform,
                                                  sigma_parms_uniform)
        sdev_cube = parm_cubes.extract('standard_deviation')[0]
        Lx_cube = parm_cubes.extract('Lx')[0]
        Ly_cube = parm_cubes.extract('Ly')[0]
        theta_cube = parm_cubes.extract('theta')[0]
        super().__init__(v, sdev_cube, Lx_cube, Ly_cube, theta_cube)


class EllipseSimulation_PrescribedParms(_EllipseSimulation):

    def __init__(self,
                 v,
                 sdev_cube,
                 Lx_cube,
                 Ly_cube,
                 theta_cube):
        super().__init__(v, sdev_cube, Lx_cube, Ly_cube, theta_cube)


def average_LxLyTheta(lx, ly, the, check_flip=False):
    '''
    Averages Lx, Ly, theta
    ans = Lx, Ly, E(theta) in deg (floats, not iris/xarray cubes)
    '''
    sigmas = []
    zipper = zip(lx.slices([]), ly.slices([]), the.slices([]))
    for lx_mini, ly_mini, the_mini in zipper:
        sigma = cube_covariance.sigma_rot_func(lx_mini.data, ly_mini.data, the_mini.data)
        sigmas.append(sigma)
    sigmas = np.array(sigmas)
    print(sigmas.shape)
    sigma_bar = np.mean(sigmas, axis=0)
    print(sigma_bar)
    eval, evec = np.linalg.eig(sigma_bar)
    E_lx = np.sqrt(eval[0])
    E_ly = np.sqrt(eval[1])
    E_the = np.arctan2(evec[1, 0], evec[0, 0])
    if check_flip:
        if E_lx < E_ly:
            moo = E_lx
            E_lx = E_ly
            E_ly = moo
            E_the += np.pi/2
    if E_the > np.pi/2:
        E_the -= np.pi
    if E_the < -np.pi/2:
        E_the += np.pi
    E_the_deg = np.rad2deg(E_the)
    return (E_lx, E_ly, E_the_deg)


def chisq_test_using_likelihood_ratios_4_covariance(sigma_hat, sigma_actual, n):
    '''
    For cases here have fixed parameters is easy
    So each grid point as well as the average estimated parameter (the estimator) 
    should be within 1-2x of the standard error of the actual parameter
    What is the likelihood of the estimator relative to the actual?
    
    For variable parms... we will just compare the regional average

    W = -n*np.log(np.linalg.det(inside_the_brackets))-n*p+n*np.trace(inside_the_brackets)
    W ~ ChiSquare(p*(p+1)/2)

    How to:
    https://www.stat.pitt.edu/sungkyu/course/2221Fall13/lec3.pdf
    https://cran.r-project.org/web//packages/mvhtests/mvhtests.pdf <<< R docs (argh)
    '''
    sigma_hat_unbiased = (n/(n-1))*sigma_hat # unbiased estimator to sigma_hat for all practical purpose n/n-1 ~ 1
    print('sigma_hat = ', sigma_hat)
    print('sigma_hat_unbiased = ', sigma_hat_unbiased)
    print('sigma_0 = ', sigma_actual)
    sigma_actual_inv = np.linalg.inv(sigma_actual)
    p = sigma_actual.shape[0] # Number of parameters (aka the dimension of the covariance) -- 2
    inside_the_brackets = sigma_actual_inv@sigma_hat_unbiased
    W = -n*np.log(np.linalg.det(inside_the_brackets)) - n*p + n*np.trace(inside_the_brackets)
    # Both Ws are the same
    # eig_inside_the_brackets = np.linalg.eigvals(inside_the_brackets)
    # alpha, g = np.mean(eig_inside_the_brackets), stats.gmean(eig_inside_the_brackets)
    # W = n*p*(alpha-np.log(g)-1)
    p_val = stats.chi2.sf(W, p*(p+1)/2)
    return (W, p_val)


def extract_LxLytheta(cubelist):
    Lx = cubelist.extract('Lx')[0]
    Ly = cubelist.extract('Ly')[0]
    theta = cubelist.extract('theta')[0]
    return (Lx, Ly, theta)


def regmean_simulated_ellipse_vs_known_regmean(fitted_parms_cubelist, actuals_parms, n_sims):
    '''
    Test against regional mean
    '''
    Lx_estimated, Ly_estimated, theta_estimated = extract_LxLytheta(fitted_parms_cubelist)
    Lx_hat, Ly_hat, theta_hat_deg = average_LxLyTheta(Lx_estimated, Ly_estimated, theta_estimated)
    theta_hat = np.deg2rad(theta_hat_deg)
    W, p_val = simulated_ellipse_vs_known_regmean([Lx_hat, Ly_hat, theta_hat],
                                                  [actuals_parms[0], actuals_parms[1], actuals_parms[2]],
                                                  n_sims)
    return (W, p_val)


def simulated_ellipse_vs_known_regmean(simulated_parms, actuals_parms, n_sims):
    '''
    Test against regional mean
    '''
    sigma_hat = cube_covariance.sigma_rot_func(simulated_parms[0], simulated_parms[1], simulated_parms[2])
    print('sigma_hat:')
    print(simulated_parms[0], simulated_parms[1], simulated_parms[2])
    print(sigma_hat)
    sigma_actual = cube_covariance.sigma_rot_func(actuals_parms[0], actuals_parms[1], actuals_parms[2])
    print('sigma_0:')
    print(actuals_parms[0], actuals_parms[1], actuals_parms[2])
    print(sigma_actual)
    print('n_sims = ', n_sims)
    W, p_val = chisq_test_using_likelihood_ratios_4_covariance(sigma_hat, sigma_actual, n_sims)
    print('W (chi-sq) likelihood ratio test stat = ', W)
    print('p-value = ', p_val)
    return (W, p_val)


def main():
    print('=== Main ===')


if __name__ == "__main__":
    main()
