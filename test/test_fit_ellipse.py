"""
pytest the ellipse estimation, stiching code
using fixed variogram parameters
"""

import pytest
# import os

import numpy as np
import iris
from iris.util import equalise_attributes
# from iris.fileformats import netcdf as inc

from ellipse_estimation import cube_covariance
from ellipse_estimation import simulate_ellipse as se_og
import test_simulate_ellipse as seut


@pytest.mark.parametrize(
    "v, cube_template, sdev, sigma_parms, size, outname",
    [
        (
            0.5,
            seut.cube_A_iris,
            0.6,
            [1000, 1000, 0],
            2500,
            "constant_1000_unrotated_0p5",
        ),
        (
            1.5,
            seut.cube_A_iris,
            0.6,
            [1000, 1000, 0],
            2500,
            "constant_1000_unrotated_1p5",
        ),
        (
            1.5,
            seut.cube_A_iris,
            0.6,
            [1300, 1100, 0],
            2500,
            "Lx1300_Ly1100_unrotated_1p5",
        ),
        (
            1.5,
            seut.cube_A_iris,
            0.6,
            [1300, 1100, np.pi / 4],
            2500,
            "Lx1300_Ly1100_45deg_1p5",
        ),
        (
            1.5,
            seut.cube_A_iris,
            0.6,
            [1500, 1000, 0],
            2500,
            "Lx1500_Ly1000_unrotated_1p5",
        ),
        (
            1.5,
            seut.cube_A_iris,
            0.6,
            [1500, 1000, np.pi / 4],
            2500,
            "Lx1500_Ly1000_45deg_1p5",
        ),
    ],
)
def test_EllipseSimulation_UniformParms_FitRandomData(
    v, cube_template, sdev, sigma_parms, size, outname
):
    """
    Unit testing the following code:
    - the stiching of Matern kernel into big covariances
    - the actual fitting of the Matern kernel
    - Stochastic simulation of multivariate norm (this is part of the perturbation function)

    1) A set of prescribed Matern covariance model and variance parameters are prescribed
    2) Construct the actual spatial covariance using the stiching code
    3) Stochastically generate size-numbered of multivariate norm field using the actual covariance
    4) Apply the fitting scheme
    5) Rebuilt a new covariance using the fitted parameter
    6) Compare this estimated covariance with the original using likelihood ratios, Wilks Theorem, and ChiSq Test
    X7) The original version of this code generates test outputs which are in ../ellipse_estimation/tests_and_examples/test_data/
    X8) A Jupyter notebook within ../ellipse_estimation/tests_and_examples/fit_tests plot those results
    """
    #
    np.random.seed(12345)
    np.set_printoptions(suppress=True)
    init_values = (900.0, 900.0, 0)
    fit_bounds = (
        (300.0, 30000.0),
        (300.0, 30000.0),
        (-2.0 * np.pi, 2.0 * np.pi),
    )
    #
    assert size < 2880, (
        "template file only has 2880 entries to copy the time coordinate"
    )
    ans = seut.test_EllipseSimulation_UniformParms(
        v, cube_template, sdev, sigma_parms
    )
    # random_GP_vectors = ans.simulate_cov(mean=0.0, size=size, reshaped_2_og=False)
    # print(random_GP_vectors)
    # print(random_GP_vectors.shape)
    random_GP_maps = ans.simulate_cov(mean=0.0, size=size, reshaped_2_og=True)
    print(random_GP_maps.shape)
    t_coord = seut.cube_B_iris.coord("time")[:size]
    random_GP_cube = ans.simulated_map_as_iris_cube_with_fake_t(
        random_GP_maps, t_coord
    )
    print(repr(random_GP_cube))
    print(random_GP_cube)
    cov_cube_instance = cube_covariance.CovarianceCube(random_GP_cube)
    print(ans.CCPLE_out.cov_ns)
    print(cov_cube_instance.Cov / size)
    #
    super_cube_list = iris.cube.CubeList()
    for _, zonal_slice in enumerate(ans.Lx[4:-4].slices(["longitude"])):
        zonal_cube_list = iris.cube.CubeList()
        for box_count, invidiual_box in enumerate(zonal_slice[4:-4].slices([])):
            current_lon = invidiual_box.coord("longitude").points[0]
            current_lat = invidiual_box.coord("latitude").points[0]
            xy, actual_latlon = (
                cov_cube_instance.find_nearest_xy_index_in_cov_matrix(
                    [current_lon, current_lat], use_full=True
                )
            )
            print(box_count, current_lat, current_lon, actual_latlon)
            kwargs = {
                "v": ans.v,
                "fform": "anistropic_rotated_pd",
                "guesses": init_values,
                "bounds": fit_bounds,
            }
            print(kwargs)
            moo = cov_cube_instance.fit_ellipse_model(xy, **kwargs)
            print(moo)
            moo_iris = moo["Model_as_1D_cube"]
            moo_H = (
                moo["Model"].x,
                moo["Model"].x[-1] * 180.0 / np.pi,
            )  # (Lx, Ly, theta (in deg))
            print(moo_H)
            for individual_ans in moo_iris:
                zonal_cube_list.append(individual_ans)
                zonal_cube_list.concatenate()
        for zonal_ans_cube in zonal_cube_list:
            super_cube_list.append(zonal_ans_cube)
            equalise_attributes(super_cube_list)
    equalise_attributes(super_cube_list)
    super_cube_list = super_cube_list.concatenate()
    vstring = str(ans.v).replace(".", "p")
    vstring = "_v_eq_" + vstring
    #
    # Check if fitted parameters are consistent with the expectation of the prescribed ones
    critical_val = 0.05
    # Check if stiched covariance based on estimated parameters consistent with the prescribed one
    print(
        "Check: Is this covariance constructed using fitted ellipse parameter consistent with the prescribed one?"
    )
    Lx_hat, Ly_hat, theta_hat, sdev_hat = se_og.extract_LxLytheta(
        super_cube_list
    )
    ans_simulated = se_og.EllipseSimulation_PrescribedParms(
        v, sdev_hat, Lx_hat, Ly_hat, theta_hat
    )
    ans_actual_subsampled = se_og.EllipseSimulation_PrescribedParms(
        v,
        ans.sdev[4:-4, 4:-4],
        ans.Lx[4:-4, 4:-4],
        ans.Ly[4:-4, 4:-4],
        ans.theta[4:-4, 4:-4],
    )
    ans_simulated.create_cov()
    ans_actual_subsampled.create_cov()
    cov_og = ans_actual_subsampled.CCPLE_out.cov_ns
    cov_hat = ans_simulated.CCPLE_out.cov_ns
    print("cov_og = ", cov_og)
    print("cov_og.shape = ", cov_og.shape)
    print("cov_hat = ", cov_hat)
    print("cov_hat.shape = ", cov_hat.shape)
    W, p_val = se_og.chisq_test_using_likelihood_ratios_4_covariance(
        cov_hat, cov_og, size
    )
    print("W (chi-sq) likelihood ratio test stat = ", W)
    print("p-value = ", p_val)
    assert p_val > critical_val, (
        "Null hypothesis not rejected at " + str(critical_val) + " level"
    )
    # Regional mean check
    print("Regional mean check")
    W, p_val = se_og.regmean_simulated_ellipse_vs_known_regmean(
        super_cube_list, (sigma_parms[0], sigma_parms[1], sigma_parms[2]), size
    )
    assert p_val > critical_val, (
        "Null hypothesis not rejected at " + str(critical_val) + " level"
    )
    #
    print("Estimated 2D kernel (ellipse parameter) looks fine.")
    #
    # Saving outputs for visualisation
    # outpath = os.path.dirname(__file__)+'/../ellipse_estimation/tests_and_examples/test_data/'
    # outncfilename = outpath+outname+'.nc'
    # print('Results to be saved...')
    # print(super_cube_list)
    # print('Saving results to ',outncfilename)
    # inc.save(super_cube_list, outncfilename)
