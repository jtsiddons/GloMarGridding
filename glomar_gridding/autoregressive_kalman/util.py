"""Common functions"""
import numpy as np


def check_sq_matrix(arr: np.ndarray) -> bool:
    """Check if arr is 2D and square"""
    return (
        len(arr.shape) == 2
        and arr.shape[0] == arr.shape[1]
    )


def check_1d(arr: np.ndarray):
    """
    Check if array a is 1D, 2D square, or something invalid
    All other arrays are banished to /dev/null inferno

    Parameters
    ----------
    arr : numpy.ndarray
        An array to be checked if it is 1D, 2D or some other dimensions

    Returns
    -------
    check: bool
        True means 1D
        False means 2D Square
        Anything else throws an exception
    """
    if len(arr.shape) == 1:
        return True
    elif len(arr.shape) == 2:
        if arr.shape[0] == arr.shape[1]:
            return False
        else:
            raise ValueError("arr is 2D but not square")
    else:
        raise ValueError("arr is not 1 or 2D")
