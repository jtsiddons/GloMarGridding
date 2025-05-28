import pytest
import numpy as np
from scipy.spatial.transform import Rotation as R
from glomar_gridding.distances import rot_mat, inv_2d


@pytest.mark.parametrize(
    "mat,",
    [
        ([1, 0, 0, 1],),
        ([1, 2, -2, 4],),
        ([7, 1, 4.22, -0.11],),
        (list(np.random.randn(4)),),
    ],
)
def test_inverse(mat):
    mat = np.asarray(mat).reshape((2, 2))
    npinv = np.linalg.inv(mat)
    inv = inv_2d(mat)

    assert np.allclose(inv, npinv)
    return None


@pytest.mark.parametrize(
    "angle,",
    [
        (np.pi / 2,),
        (0.123,),
        (-np.pi / 3,),
        (np.pi / 12,),
    ],
)
def test_rot(angle):
    rot = rot_mat(angle)
    r = np.asarray(R.from_rotvec(angle * np.array([0, 0, 1])).as_matrix())[
        :2, :2
    ]
    print(f"{rot = }")
    print(f"{r= }")

    assert np.allclose(r, rot)
