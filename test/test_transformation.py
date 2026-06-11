import numpy as np

from glomar_gridding.transformation import (
    weibull_to_normality,
    inv_weibull_to_normality,
)


def test_weibull() -> None:
    """African or European?"""
    fast_swallow = 16.0
    slow_swallow = 11.0
    speeds = [slow_swallow, fast_swallow]
    shapes = [np.e, np.pi]
    for speed in speeds:
        for shape in shapes:
            for use_kp11 in [True, False]:
                transformed = weibull_to_normality(
                    speed,
                    shape,
                    use_kp11=use_kp11,
                )
                inverse = inv_weibull_to_normality(
                    transformed,
                    shape,
                    use_kp11=use_kp11,
                )
                assert np.isclose(inverse, speed)
