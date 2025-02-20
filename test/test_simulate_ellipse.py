import pytest
import os

import iris
import xarray as xa
import numpy as np

from ellipse_estimation import simulate_ellipse as s_e

nc_A = os.path.dirname(__file__)+'/../ellipse_estimation/tests_and_examples/test_data/blank_cube.nc'
nc_B = os.path.dirname(__file__)+'/../ellipse_estimation/tests_and_examples/test_data/data_-170_-10.nc'
cube_A_iris = iris.load_cube(nc_A)
cube_A_xa = xa.load_dataset(nc_A)['tas']
cube_B_iris = iris.load_cube(nc_B, 'sst_anomalies_projected_to_top_10_global_pcas')
cube_B_xa = xa.load_dataset(nc_B)['sst_anomalies_projected_to_top_10_global_pcas']

@pytest.mark.parametrize(
    "cube, sdev, sigma_parms",
    [
        (cube_A_iris, 0.6, [1500, 1000, np.pi/8]), # Template is an iris cube
        (cube_A_xa, 1.2, [1500, 1000, np.pi/8]), # Template is a xarray data array
        # (cube_A_iris, 'tortoise', [1500, 1000, np.pi/8]), # This will fail
        # (cube_A_iris, 0.6, [cube_A_xa, 1000, np.pi/8]), # This will fail
    ]
)
def test_fill_cube_with_uniform_parms(cube, sdev, sigma_parms):
    ans = s_e.fill_cube_with_uniform_parms(cube, sdev, sigma_parms)
    assert isinstance(ans, iris.cube.CubeList)
    for ebuc in ans:
        print(repr(ebuc))
        print(ebuc.data[0, :5], ebuc.data[-1, -5:])


@pytest.mark.parametrize(
    "v, cube_template, sdev, sigma_parms",
    [
        (0.5, cube_A_iris, 0.6, [1500, 1000, 0]), # Template is an iris cube
        (1.5, cube_A_iris, 0.6, [1500, 1000, np.pi/8]), # Template is an iris cube
    ]
)
def test_EllipseSimulation_UniformParms(v, cube_template, sdev, sigma_parms):
    print('This is test_EllipseSimulation_UniformParms')
    ans = s_e.EllipseSimulation_UniformParms(v, cube_template, sdev, sigma_parms)
    print(repr(ans))
    ans.create_cov()
    print(ans.CCPLE_out.cor_ns)
    print(ans.CCPLE_out.cov_ns)
    if (v == 0.5) and (sigma_parms[-1] == 0.0):
        pt0 = cube_template[0, 0]
        pt1 = cube_template[0, 1]
        y = pt0.coord('latitude').points[0]
        x0 = pt0.coord('longitude').points[0]
        x1 = pt1.coord('longitude').points[0]
        print(pt0)
        print(pt1)
        ## tau = 1 (e.g. 1 Lx, 0 Ly) ---> correlation = np.exp(-np.sqrt(2))
        dist = (x1 - x0)*60.0*1.852*np.cos(np.deg2rad(y))
        corr_1st_two = ans.CCPLE_out.cor_ns[:2, :2]
        exp_predicted_val = np.exp(-np.sqrt(2)*dist/sigma_parms[0])
        print('Below uses d/2 convention -- "Karspeck"')
        print('Verbal check if the correlation is correct')
        print('If v=0.5, corr(tau=1) = np.exp(-SQRT(2))')
        print('Distance between x0 and x1 = ', dist)
        print('Effective Range = ', sigma_parms[0])
        print('tau = ', dist/sigma_parms[0])
        print('Ellipse code/Matern predicted correlation Matern = ', corr_1st_two[0, 1])
        print('Exponential predicted correlation = ', exp_predicted_val)
        print('Diff = ', exp_predicted_val-corr_1st_two[0, 1])
    return ans


Atlantic_parm_cube = os.path.dirname(__file__)+'/../ellipse_estimation/tests_and_examples/test_data/Atlantic_Ocean_07.nc'
cube_matern_dist = iris.load(Atlantic_parm_cube)
Atlantic_Lx = cube_matern_dist.extract('Lx')[0][50:70, 50:70]
Atlantic_Ly = cube_matern_dist.extract('Ly')[0][50:70, 50:70]
Atlantic_theta = cube_matern_dist.extract('theta')[0][50:70, 50:70]
Atlantic_sigma = cube_matern_dist.extract('standard_deviation')[0][50:70, 50:70]
Atlantic_v = Atlantic_Lx.coord('v_shape').points[0]

@pytest.mark.parametrize(
    "v, sdev_cube, Lx_cube, Ly_cube, theta_cube",
    [
        (Atlantic_v, Atlantic_sigma, Atlantic_Lx, Atlantic_Ly, Atlantic_theta),
    ]
)
def test_EllipseSimulation_PrescribedParms(v, sdev_cube, Lx_cube, Ly_cube, theta_cube):
    print('This is test_EllipseSimulation_PrescribedParms')
    ans = s_e.EllipseSimulation_PrescribedParms(v,
                                                sdev_cube,
                                                Lx_cube,
                                                Ly_cube,
                                                theta_cube)
    print(repr(ans))
    return ans


@pytest.mark.parametrize(
    "v, sdev_func, Lx_func, Ly_func, theta_func",
    [
        # (1.5,
        #  lambda x, y: 1.2+0.*x+0.*y,
        #  lambda x, y: 1500.+100.*x+0.*y,
        #  lambda x, y: 500.+0.*x+50.*y,
        #  lambda x, y: 0+0.*x+0.*y),
        (1.5,
         lambda x, y: 0.6+0.05*x+0.01*y,
         lambda x, y: 1500.+100.*x+0.*y,
         lambda x, y: 500.+0.*x+50.*y,
         lambda x, y: 0.+0.*x+0.*y),
        # (1.5,
        #  lambda x, y: 1.2+0.*x+0.*y,
        #  lambda x, y: 1500.+100.*x+0.*y,
        #  lambda x, y: 500.+0.*x+50.*y,
        #  lambda x, y: np.pi/4+0.*x+0.*y),
    ]
)
def test_EllipseSimulation_NonStatFunc(v, sdev_func, Lx_func, Ly_func, theta_func):
    print('This is test_EllipseSimulation_NonStatFunc')
    nlons = cube_A_iris.coord('longitude').shape[0]
    nlats = cube_A_iris.coord('latitude').shape[0]
    ii, jj = np.meshgrid(np.arange(nlons), np.arange(nlats))
    def compute_parms(func, meta_template):
        out_cube = cube_A_iris.copy()
        out_cube.data = np.ma.masked_where(False, func(ii, jj))
        out_cube.rename(meta_template.name())
        out_cube.units = meta_template.units
        return out_cube
    template_sigma = compute_parms(sdev_func, Atlantic_sigma)
    template_Lx = compute_parms(Lx_func, Atlantic_Lx)
    template_Ly = compute_parms(Ly_func, Atlantic_Ly)
    template_theta = compute_parms(theta_func, Atlantic_theta)
    ans = s_e.EllipseSimulation_PrescribedParms(v,
                                                template_sigma,
                                                template_Lx,
                                                template_Ly,
                                                template_theta)
    ans.create_cov()
    np.set_printoptions(suppress=True)
    print(repr(ans))
    print(ans.CCPLE_out.cor_ns)
    print(ans.CCPLE_out.cov_ns)
    return ans
