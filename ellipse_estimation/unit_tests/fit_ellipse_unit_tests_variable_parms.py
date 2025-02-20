import pytest

import numpy as np
import iris
from iris.util import equalise_attributes
from iris.fileformats import netcdf as inc
# from statsmodels.stats.multitest import multipletests

from ellipse_estimation import cube_covariance
from ellipse_estimation import simulate_ellipse as se_og
from nonstationary_cov.unit_tests import simulate_ellipse_unit_tests as seut

@pytest.mark.parametrize(
    "v, sdevLxLyTheta_func, size, outname",
    [
        (1.5,
         [lambda x, y: 0.6+0.0*x+0.0*y,
          lambda x, y: 1500.+100.*x+0.*y,
          lambda x, y: 500.+0.*x+50.*y,
          lambda x, y: 0.+0.*x+0.*y],
         2500,
         "variable_scales_1p5_fixed_sdev_fixed_angle"),
        (1.5,
         [lambda x, y: 0.6+0.05*x+0.01*y,
          lambda x, y: 1500.+100.*x+0.*y,
          lambda x, y: 500.+0.*x+50.*y,
          lambda x, y: 0.+0.*x+0.*y],
         2500,
         "variable_scales_1p5_variable_sdev_fixed_angle"),
        (1.5,
         [lambda x, y: 0.6+0.05*x+0.01*y,
          lambda x, y: 1500.+100.*x+0.*y,
          lambda x, y: 500.+0.*x+50.*y,
          lambda x, y: 0.+0.*x+np.pi/4/16.*y],
         2500,
         "variable_scales_1p5_variable_sdev_variable_angle"),
    ]
)
def test_EllipseSimulation_VariableParms_FitRandomData(v,
                                                       sdevLxLyTheta_func,
                                                       size,
                                                       outname):
    #
    np.set_printoptions(suppress=True)
    init_values = (900.0, 900.0, 0)
    fit_bounds = ((300.0, 30000.0),
                  (300.0, 30000.0),
                  (-2.0*np.pi, 2.0*np.pi))
    #
    assert size < 2880, 'template file only has 2880 entries to copy the time coordinate'
    ans = seut.test_EllipseSimulation_NonStatFunc(v,
                                                  sdevLxLyTheta_func[0],
                                                  sdevLxLyTheta_func[1],
                                                  sdevLxLyTheta_func[2],
                                                  sdevLxLyTheta_func[3])
    random_GP_maps = ans.simulate_cov(mean=0.0, size=size, reshaped_2_og=True)
    print(random_GP_maps.shape)
    t_coord = seut.cube_B_iris.coord('time')[:size]
    random_GP_cube = ans.simulated_map_as_iris_cube_with_fake_t(random_GP_maps, t_coord)
    print(repr(random_GP_cube))
    print(random_GP_cube)
    cov_cube_instance = cube_covariance.CovarianceCube(random_GP_cube)
    print(ans.CCPLE_out.cov_ns)
    print(cov_cube_instance.Cov/size)
    #
    super_cube_list = iris.cube.CubeList()
    for zonal, zonal_slice in enumerate(ans.Lx[4:-4].slices(['longitude'])):
        zonal_cube_list = iris.cube.CubeList()
        for box_count, invidiual_box in enumerate(zonal_slice[4:-4].slices([])):
            current_lon = invidiual_box.coord('longitude').points[0]
            current_lat = invidiual_box.coord('latitude').points[0]
            xy, actual_latlon = cov_cube_instance.find_nearest_xy_index_in_cov_matrix([current_lon, current_lat], use_full=True)
            print(box_count, current_lat, current_lon,actual_latlon)
            kwargs = {'v': ans.v,
                      'fform': 'anistropic_rotated_pd',
                      'guesses': init_values,
                      'bounds': fit_bounds}
            print(kwargs)
            moo = cov_cube_instance.ps2006_kks2011_model(xy, **kwargs)
            print(moo)
            moo_iris = moo['Model_as_1D_cube']
            moo_H = (moo['Model'].x, moo['Model'].x[-1]*180.0/np.pi) # (Lx, Ly, theta (in deg))
            print(moo_H)
            for individual_ans in moo_iris:
                zonal_cube_list.append(individual_ans)
                zonal_cube_list.concatenate()
        for zonal_ans_cube in zonal_cube_list:
            super_cube_list.append(zonal_ans_cube)
            equalise_attributes(super_cube_list)
    equalise_attributes(super_cube_list)
    super_cube_list = super_cube_list.concatenate()
    vstring = str(ans.v).replace('.','p')
    vstring = '_v_eq_'+vstring
    #
    # Check if fitted parameters are consistent with the expectation of the prescribed ones
    actual_Lx = ans.Lx[4:-4]
    actual_Ly = ans.Ly[4:-4]
    actual_theta = ans.theta[4:-4]
    critical_val = 0.05
    # Regional mean check
    print('Regional mean check')
    sigma_parms_ = se_og.average_LxLyTheta(actual_Lx, actual_Ly, actual_theta)
    sigma_parms = [sigma_parms_[0], sigma_parms_[1], np.deg2rad(sigma_parms_[-1])]
    print(sigma_parms)
    W, p_val = se_og.regmean_simulated_ellipse_vs_known_regmean(super_cube_list,
                                                                (sigma_parms[0], sigma_parms[1], sigma_parms[2]),
                                                                size)
    print(W, p_val)
    assert p_val > critical_val, "Null hypothesis not rejected at "+str(critical_val)+" level or you are one really (un)lucky person"
    # # Check at each point with multi-testing correction
    # print('Local checks?')
    # Lx_hat, Ly_hat, theta_hat = se_og.extract_LxLytheta(super_cube_list)
    # Ws, p_vals = [], []
    # zipper = zip(Lx_hat.slices([]), Ly_hat.slices([]), theta_hat.slices([]),
    #              actual_Lx.slices([]), actual_Ly.slices([]), actual_theta.slices([]))
    # for Lx_hat_cubbie, Ly_hat_cubbie, theta_hat_cubbie, actual_Lx_cubbie, actual_Ly_cubbie, actual_theta_cubbie in zipper:
    #     sigma_hat_parms = (Lx_hat_cubbie.data, Ly_hat_cubbie.data, theta_hat_cubbie.data)
    #     sigma_actual_parms = (actual_Lx_cubbie.data, actual_Ly_cubbie.data, actual_theta_cubbie.data)
    #     ans = se_og.simulated_ellipse_vs_known_regmean(sigma_hat_parms, sigma_actual_parms, size)
    #     Ws.append(ans[0])
    #     p_vals.append(ans[1])
    # Ws = np.array(Ws)
    # p_vals = np.array(p_vals)
    # print(Ws)
    # print(p_vals)
    # fdr_ans = multipletests(p_vals, alpha=critical_val, method='fdr_bh')
    # print(fdr_ans)
    # p_vals_adjs = fdr_ans[1]
    # assert np.all(p_vals_adjs > critical_val)
    #
    print('Estimated 2D kernel (ellipse parameter) looks fine.')
    #
    outpath = '../tests_and_examples/test_data/'
    outncfilename = outpath+outname+'.nc'
    print('Results to be saved...')
    print(super_cube_list)
    print('Saving results to ',outncfilename)
    inc.save(super_cube_list, outncfilename)
