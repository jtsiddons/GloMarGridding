"""
Functions for working with kriging outputs or arrays. For example merging two
arrays using a set of weights.
"""

import xarray as xr


def merge_by_weights(
    left: xr.DataArray,
    right: xr.DataArray,
    weights: xr.DataArray,
    output_name: str = "combined",
) -> xr.DataArray:
    """
    Merge two DataArrays using a third DataArray that contains values between
    0 and 1 that indicate the the proportion of the first array, and 1 - the
    weight value indicates the proportion of the 2nd array.

        result = left * weights + (1 - weights) * right

    An example use-case would be to combine an array of ocean observations with
    an array of land observations, the weights would indicate the proportion of
    ocean within a given grid-box.

    The input arrays must all have the same shape.

    Parameters
    ----------
    left : xarray.DataArray
        The first array, this array contains values corresponding directly to
        the weights values as the amount of the array to use in the merge.
    right : xarray.DataArray
        The second array, this array contains values corresponding  to
        1 - the weights values as the amount of the array to use in the merge.
    weights : xarray.DataArray
        The weights, values between 0 and 1 indicating the proportion of `left`
        to use in the resulting combined array. 1 - weights is the proportion
        of `right`.
    output_name : str
        Name of the output array

    Returns
    -------
    combined : xarray.DataArray
        The two input arrays combined using the weights as a weighting.
    """
    if not (left.shape == right.shape == weights.shape):
        raise ValueError("All input arrays must have the same shape")
    combined = (left * weights) + ((1 - weights) * right)
    combined.name = output_name
    return combined
