import pytest
import os

import iris

from ellipse_estimation import distance_util as d_u

nc_A = os.path.dirname(__file__)+'/../ellipse_estimation/tests_and_examples/test_data/blank_cube.nc'
cube_A_iris = iris.load_cube(nc_A)

'''
Use pytest -s
'''

@pytest.mark.parametrize(
    "cube_i_indices, cube_j_indices",
    [
        ([0, 0], [-1,-1]),
        ([1, 1], [-2,-2]),
        ([1, 2], [-3,-4]),
        ([5, 5], [5, 5]),
    ]
)
def test_distance(cube_i_indices, cube_j_indices):
    '''
    pytest scalar_cube_great_circle_distance_cube and scalar_cube_great_circle_distance_
    '''
    cube_i = cube_A_iris[cube_i_indices[0], cube_i_indices[1]]
    cube_j = cube_A_iris[cube_j_indices[0], cube_j_indices[1]]
    print([cube_i.coord('latitude').points[0], cube_i.coord('longitude').points[0]])
    print([cube_j.coord('latitude').points[0], cube_j.coord('longitude').points[0]])
    ans = d_u.scalar_cube_great_circle_distance_cube(cube_i, cube_j)
    print(ans)
