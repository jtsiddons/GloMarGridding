# Copyright 2025 National Oceanography Centre
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Functions for computing correlated and uncorrelated components of the error
covariance. These values are determined from standard deviation (sigma) values
assigned to groupings within the observational data.

The correlated components will form a matrix that is permutationally equivalent
to a block diagonal matrix (i.e. the matrix will be block diagonal if the
observational data is sorted by the group).

The uncorrelated components will form a diagonal matrix.

Further a distance-based component can be constructed, where distances between
records within the same grid box are evaluated.

The functions in this module are valid for observational data where there could
be more than 1 observation in a gridbox.
"""

from collections.abc import Callable
from warnings import warn

import numpy as np
import polars as pl

from .utils import ColumnNotFoundError, check_cols


def uncorrelated_components(
    df: pl.DataFrame,
    group_col: str = "data_type",
    obs_sig_col: str | None = None,
    obs_sig_map: dict[str, float] | None = None,
) -> np.ndarray:
    """
    Calculates the covariance matrix of the measurements (observations). This
    is the uncorrelated component of the covariance.

    The result is a diagonal matrix. The diagonal is formed by the square of the
    sigma values associated with the values in the grouping.

    The values can either be pre-defined in the observational dataframe, and
    can be indicated by the "bias_val_col" argument. Alternatively, a mapping
    can be passed, the values will be then assigned by this mapping of group to
    sigma.

    Parameters
    ----------
    df : polars.DataFrame
        The observational DataFrame containing values to group by.
    group_col : str
        Name of the group column to use to set observational sigma values.
    obs_sig_col : str | None
        Name of the column containing observational sigma values. If set and
        present in the DataFrame, then this column is used as the diagonal of
        the returned covariance matrix.
    obs_sig_map : dict[str, float] | None
        Mapping between group and observational sigma values used to define
        the diagonal of the returned covariance matrix.

    Returns
    -------
    A diagonal matrix representing the uncorrelated components of the error
    covariance matrix.
    """
    if obs_sig_col is not None and obs_sig_col in df.columns:
        return np.diag(df.get_column(obs_sig_col))
    elif obs_sig_col is not None and obs_sig_col not in df.columns:
        raise ColumnNotFoundError(
            f"Observation Bias Column {obs_sig_col} not found."
        )

    obs_sig_map = obs_sig_map or {}
    groupings: pl.Series = df.get_column(group_col)
    s = groupings.replace_strict(
        {k: v**2 for k, v in obs_sig_map.items()}, default=0.0
    )
    if s.eq(0.0).all():
        warn("No values in obs_covariance set")
    elif s.eq(0.0).any():
        warn("Some values in obs_covariance not set")

    return np.diag(s)


def correlated_components(
    df: pl.DataFrame,
    group_col: str,
    bias_sig_col: str | None = None,
    bias_sig_map: dict[str, float] | None = None,
) -> np.ndarray:
    """
    Returns measurements covariance matrix updated by adding bias uncertainty to
    the measurements based on a grouping within the observational data.

    The result is equivalent to a block diagonal matrix via permutation. If the
    input observational data is sorted by the group column then the resulting
    matrix is block diagonal, where the blocks are the size of each grouping.
    The values in each block are the square of the sigma value associated with
    the grouping.

    Note that in most cases the output is not a block-diagonal, as the input
    is not usually sorted by the group column. In most processing cases, the
    input dataframe will be sorted by the gridbox index.

    The values can either be pre-defined in the observational dataframe, and
    can be indicated by the "bias_val_col" argument. Alternatively, a mapping
    can be passed, the values will be then assigned by this mapping of group to
    sigma.

    Parameters
    ----------
    df : polars.DataFrame
        Observational DataFrame including group information and bias uncertainty
        values for each grouping. It is assumed that a single bias uncertainty
        value applies to the whole group, and is applied as cross terms in the
        covariance matrix (plus to the diagonal).
    group_col : str
        Name of the column that can be used to partition the observational
        DataFrame.
    bias_sig_col : str | None
        Name of the column containing bias uncertainty values for each of
        the groups identified by 'group_col'. It is assumed that a single bias
        uncertainty value applies to the whole group, and is applied as cross
        terms in the covariance matrix (plus to the diagonal).
    bias_sig_map : dict[str, float] | None
        Mapping between values in the group_col and bias uncertainty values,
        if bias_val_col is not in the DataFrame.

    Returns
    -------
    The correlated components of the error covariance.
    """
    check_cols(df, [group_col])

    # Initialise array
    covx = np.zeros((len(df), len(df)))

    bias_sig_col = bias_sig_col or "_bias_uncert"
    bias_sig_map = bias_sig_map or {}

    if bias_sig_col not in df.columns:
        df = df.with_columns(
            pl.col(group_col)
            .replace_strict(
                {k: v**2 for k, v in bias_sig_map.items()},
                default=0.0,
            )
            .alias(bias_sig_col)
        )
        if df[bias_sig_col].eq(0.0).all():
            warn("No bias uncertainty values set")
        elif df[bias_sig_col].eq(0.0).any():
            warn("Some bias uncertainty values not set")

    # NOTE: polars is easier for this analysis!
    df = (
        df.select(group_col, bias_sig_col)
        .with_row_index("index")
        .group_by(group_col)
        # NOTE: It is expected that the bias value should be the same for all
        #       records within the same group
        .agg(pl.col("index"), pl.col(bias_sig_col).first())
    )
    for row in df.rows(named=True):
        if row[bias_sig_col] is None:
            print(f"Group {row[group_col]} has no bias uncertainty value set")
            continue
        # INFO: Adding cross-terms to covariance
        inds = np.ix_(row["index"], row["index"])
        covx[inds] = covx[inds] + row[bias_sig_col]

    return covx


def dist_weight(
    df: pl.DataFrame,
    dist_fn: Callable,
    grid_idx: str = "grid_idx",
    **dist_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the distance and weight matrices over gridboxes for an input Frame.

    This function acts as a wrapper for a distance function, allowing for
    computation of the distances between positions in the same gridbox using any
    distance metric.

    The weightings from this function are for the gridbox mean of the
    observations within a gridbox.

    Parameters
    ----------
    df : polars.DataFrame
        The observation DataFrame, containing the columns required for
        computation of the distance matrix. Contains the "grid_idx" column which
        indicates the gridbox for a given observation. The index of the
        DataFrame should match the index ordering for the output distance
        matrix/weights.
    dist_fn : Callable
        The function used to compute a distance matrix for all points in a given
        grid-cell. Takes as input a polars.DataFrame as first argument. Any
        other arguments should be constant over all gridboxes, or can be a
        look-up table that can use values in the DataFrame to specify values
        specific to a gridbox. The function should return a numpy matrix, which
        is the distance matrix for the gridbox only. This wrapper function will
        correctly apply this matrix to the larger distance matrix using the
        index from the DataFrame.

        If dist_fn is None, then no distances are computed and None is returned
        for the dist value.
    grid_idx : str
        Name of the column containing the grid index values
    **dist_kwargs
        Arguments to be passed to dist_fn. In general these should be constant
        across all gridboxes. It is possible to pass a look-up table that
        contains pre-computed values that are gridbox specific, if the keys can
        be matched to a column in df.

    Returns
    -------
    dist : numpy.matrix
        The distance matrix, which contains the same number of rows and columns
        as rows in the input DataFrame df. The values in the matrix are 0 if the
        indices of the row/column are for observations from different gridboxes,
        and non-zero if the row/column indices fall within the same gridbox.
        Consequently, with appropriate re-arrangement of rows and columns this
        matrix can be transformed into a block-diagonal matrix. If the DataFrame
        input is pre-sorted by the gridbox column, then the result is a
        block-diagonal matrix.

        If dist_fn is None, then this value will be None.
    weights : numpy.matrix
        A matrix of weights. This has dimensions n x p where n is the number of
        unique gridboxes and p is the number of observations (the number of rows
        in df). The values are 0 if the row and column do not correspond to the
        same gridbox and equal to the inverse of the number of observations in a
        gridbox if the row and column indices fall within the same gridbox. The
        rows of weights are in a sorted order of the gridbox. Should this be
        incorrect, one should re-arrange the rows after calling this function.
    """
    # QUESTION: Do we want to sort the unique grid-cell values?
    #           Ensures consistency between runs if the frame ordering gets
    #           shuffled in some way.
    # QUESTION: Maybe sort by "flattened_idx", then no need to sort obs?
    gridboxes = sorted(df[grid_idx].unique())
    _n_gridboxes = len(gridboxes)
    _n_obs = df.height

    df = df.with_row_index("_index")

    # Initialise
    weights = np.zeros((_n_gridboxes, _n_obs))
    dist = np.zeros((_n_obs, _n_obs))

    for i, gridbox_df in enumerate(df.partition_by(grid_idx)):
        gridbox_idcs = gridbox_df.get_column("_index").to_list()
        idcs_array = np.ix_(gridbox_idcs, gridbox_idcs)

        weights[i, gridbox_idcs] = 1 / gridbox_df.height
        dist[idcs_array] = dist_fn(gridbox_df, **dist_kwargs)

    return dist, weights


def get_weights(
    df: pl.DataFrame,
    grid_idx: str = "grid_idx",
) -> np.ndarray:
    """
    Get just the weight matrices over gridboxes for an input Frame.

    The weightings from this function are for the gridbox mean of the
    observations within a gridbox.

    Parameters
    ----------
    df : polars.DataFrame
        The observation DataFrame, containing the columns required for
        computation of the distance matrix. Contains the "grid_idx" column which
        indicates the gridbox for a given observation. The index of the
        DataFrame should match the index ordering for the output weights.
    grid_idx : str
        Name of the column containing the gridbox index from the output grid.

    Returns
    -------
    weights : numpy.matrix
        A matrix of weights. This has dimensions n x p where n is the number of
        unique gridboxes and p is the number of observations (the number of rows
        in df). The values are 0 if the row and column do not correspond to the
        same gridbox and equal to the inverse of the number of observations in a
        gridbox if the row and column indices fall within the same gridbox. The
        rows of weights are in a sorted order of the gridbox. Should this be
        incorrect, one should re-arrange the rows after calling this function.
    """
    weights = (
        df.with_row_index("_index")
        .with_columns((1 / pl.len().over(grid_idx)).alias("_weight"))
        .select(["_index", grid_idx, "_weight"])
        .pivot(on=grid_idx, index="_index", values="_weight")
        .fill_null(0)
        .sort("_index")
        .drop("_index")
    )
    return (
        weights.select(sorted(weights.columns, key=int)).to_numpy().transpose()
    )
