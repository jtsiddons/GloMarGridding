import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R
from sklearn.metrics.pairwise import haversine_distances

from glomar_gridding.distances import displacements, inv_2d, rot_mat
from glomar_gridding.ellipse.covariance import (
    _det_22_multi,
    _det_22_single,
    _haversine_multi,
    _haversine_single,
    _mo_disp_multi,
    _mo_disp_single,
    _mod_mo_disp_multi,
    _mod_mo_disp_single,
)


@pytest.mark.parametrize(
    "mat",
    [
        [1, 0, 0, 1],
        [1, 2, -2, 4],
        [7, 1, 4.22, -0.11],
        list(np.random.randn(4)),
    ],
)
def test_inverse_and_det(mat):
    print(mat)
    det = _det_22_single(mat)
    mat = np.asarray(mat).reshape((2, 2))
    npdet = np.linalg.det(mat)
    npinv = np.linalg.inv(mat)
    inv = inv_2d(mat)

    assert np.allclose(det, npdet)
    assert np.allclose(inv, npinv)

    return None


def test_det_multi():
    mats = [
        np.asarray([1, 0, 0, 1]),
        np.asarray([1, 2, -2, 4]),
        np.asarray([7, 1, 4.22, -0.11]),
        np.random.randn(4),
    ]
    npdets = np.asarray([np.linalg.det(mat.reshape((2, 2))) for mat in mats])
    mats = np.vstack(mats)
    dets = _det_22_multi(mats)

    assert np.allclose(dets, npdets)


@pytest.mark.parametrize(
    "angle,",
    [
        np.pi / 2,
        0.123,
        -np.pi / 3,
        np.pi / 12,
    ],
)
def test_rot(angle):
    rot = rot_mat(angle)
    r = np.asarray(R.from_rotvec(angle * np.array([0, 0, 1])).as_matrix())[
        :2, :2
    ]

    assert np.allclose(r, rot)


LATS0 = 90 - 180 * np.random.rand(4)
LONS0 = 180 - 360 * np.random.rand(4)
LATS1 = 90 - 180 * np.random.rand(4)
LONS1 = 180 - 360 * np.random.rand(4)

print(LATS1)
print(np.vstack([LATS1, LONS1]))

HAVERSINE_RES = 6371 * np.diag(
    haversine_distances(
        np.deg2rad(np.vstack([LATS0, LONS0])).T,
        np.deg2rad(np.vstack([LATS1, LONS1])).T,
    )
)


def get_displacements(method):
    dy, dx = displacements(LATS0, LONS0, LATS1, LONS1, delta_x_method=method)
    return list(zip(np.diag(dy) * 6371, np.diag(dx) * 6371))


MOD_MO_RES = get_displacements("Modified_Met_Office")
MO_RES = get_displacements("Met_Office")


def test_multi():
    lats0 = np.deg2rad(LATS0)
    lons0 = np.deg2rad(LONS0)
    lats1 = np.deg2rad(LATS1)
    lons1 = np.deg2rad(LONS1)

    hav_res = _haversine_multi(
        lats0,
        lons0,
        lats1,
        lons1,
    )
    assert np.allclose(hav_res, HAVERSINE_RES)

    mod_mo_res = np.asarray(_mod_mo_disp_multi(lats0, lons0, lats1, lons1))
    assert np.allclose(mod_mo_res.T, MOD_MO_RES)

    mo_res = np.asarray(_mo_disp_multi(lats0, lons0, lats1, lons1))
    print(f"{mo_res.T = }")
    print(f"{np.asarray(MO_RES) = }")
    assert np.allclose(mo_res.T, MO_RES)

    return None


@pytest.mark.parametrize("i", list(range(4)))
def test_single(i):
    lat0 = float(np.deg2rad(LATS0[i]))
    lon0 = float(np.deg2rad(LONS0[i]))
    lat1 = float(np.deg2rad(LATS1[i]))
    lon1 = float(np.deg2rad(LONS1[i]))

    hav_res = _haversine_single(lat0, lon0, lat1, lon1)
    assert np.allclose(hav_res, HAVERSINE_RES[i])

    mod_mo_res = _mod_mo_disp_single(lat0, lon0, lat1, lon1)
    assert np.allclose(mod_mo_res, MOD_MO_RES[i])

    mo_res = _mo_disp_single(lat0, lon0, lat1, lon1)
    assert np.allclose(mo_res, MO_RES[i])

    return None
