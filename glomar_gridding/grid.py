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

"""Functions for creating grids and mapping observations to a grid"""

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import polars as pl
import xarray as xr

from .distances import calculate_distance_matrix, haversine_distance_from_frame
from .utils import filter_bounds, find_nearest, select_bounds


def map_to_grid(
    obs: pl.DataFrame,
    grid: xr.DataArray,
    obs_coords: list[str] = ["lat", "lon"],
    grid_coords: list[str] = ["latitude", "longitude"],
    sort: bool = True,
    bounds: list[tuple[float, float]] | None = None,
    add_grid_pts: bool = True,
    grid_prefix: str = "grid_",
) -> pl.DataFrame:
    """
    Align an observation dataframe to a grid defined by an xarray DataArray.

    Maps observations to the nearest grid-point, and sorts the data by the
    1d index of the DataArray in a row-major format.

    The grid defined by the latitude and longitude coordinates of the input
    DataArray is then used as the output grid of the Gridding process.

    Parameters
    ----------
    obs : polars.DataFrame
        The observational DataFrame containing positional data with latitude,
        longitude values within the `obs_latname` and `obs_lonname` columns
        respectively. Observations are mapped to the nearest grid-point in the
        grid.
    grid : xarray.DataArray
        Contains the grid coordinates to map observations to.
    obs_coords : list[str]
        Names of the column containing positional values in the input
        observational DataFrame.
    grid_coords : list[str]
        Names of the coordinates in the input grid DataArray used to define the
        grid.
    sort : bool
        Sort the observational DataFrame by the grid index
    bounds : list[tuple[float, float]] | None
        Optionally filter the grid and DataFrame to fall within spatial bounds.
        This list must have the same size and ordering as `obs_coords` and
        `grid_coords` arguments.
    add_grid_pts : bool
        Add the grid positional information to the observational DataFrame.
    grid_prefix : str
        Prefix to use for the new grid columns in the observational DataFrame.

    Returns
    -------
    obs : pandas.DataFrame
        Containing additional `grid_*`, and `grid_idx` values
        indicating the positions and grid index of the observation
        respectively. The DataFrame is also sorted (ascendingly) by the
        `grid_idx` columns for consistency with the gridding functions.

    Examples
    --------
    >>> obs = pl.read_csv("/path/to/obs.csv")
    >>> grid = grid_from_resolution(
            resolution=5,
            bounds=[(-87.5, 90), (-177.5, 180)],  # Lower bound is centre
            coord_names=["lat", "lon"]
        )
    >>> obs = map_to_grid(obs, grid, grid_coords=["lat", "lon"])
    """
    if bounds is not None:
        grid = select_bounds(grid, bounds, grid_coords)
        obs = filter_bounds(obs, bounds, obs_coords)

    grid_size = grid.shape

    grid_idx: list[list[int]] = []
    obs_to_grid_pos: list[np.ndarray] = []
    for grid_coord, obs_coord in zip(grid_coords, obs_coords):
        grid_pos = grid.coords[grid_coord].values
        _grid_idx, _obs_to_grid_pos = find_nearest(grid_pos, obs[obs_coord])
        grid_idx.append(_grid_idx)
        obs_to_grid_pos.append(_obs_to_grid_pos)
        del _grid_idx, _obs_to_grid_pos

    flattened_idx = np.ravel_multi_index(
        grid_idx,
        grid_size,
        order="C",  # row-major
    )

    obs = obs.with_columns(pl.Series(grid_prefix + "idx", flattened_idx))
    if add_grid_pts:
        obs = obs.with_columns(
            [
                pl.Series(grid_prefix + obs_coord, grid_pos)
                for grid_pos, obs_coord in zip(obs_to_grid_pos, obs_coords)
            ]
        )

    if sort:
        obs = obs.sort("grid_idx", descending=False)

    return obs


def grid_from_resolution(
    resolution: float | list[float],
    bounds: list[tuple[float, float]],
    coord_names: list[str],
) -> xr.DataArray:
    """
    Generate a grid from a resolution value, or a list of resolutions for
    given boundaries and coordinate names.

    Note that all list inputs must have the same length, the ordering of values
    in the lists is assumed align.

    The constructed grid will be regular, in the sense that the grid spacing is
    constant. However, the resolution in each direction can be different,
    allowing for finer resolution in some direction.

    Parameters
    ----------
    resolution : float | list[float]
        Resolution of the grid. Can be a single resolution value that will be
        applied to all coordinates, or a list of values mapping a resolution
        value to each of the coordinates.
    bounds : list[tuple[float, float]]
        A list of bounds of the form `(lower_bound, upper_bound)` indicating
        the bounding box of the returned grid. Typically, one would set the
        lower bound to be the centre of the first grid box. The upper bound is
        an open bound (similar to usage in `range`). For example a 5 degree
        resolution longitude range between -180, 180 could be defined with
        bounds `(-177.5, 180)`.
    coord_names : list[str]
        List of coordinate names in the same order as the bounds and
        resolution(s).

    Returns
    -------
    grid : xarray.DataArray:
        The grid defined by the resolution and bounding box.

    Examples
    --------
    >>> grid_from_resolution(
            resolution=5,
            bounds=[(-87.5, 90), (-177.5, 180)],  # Lower bound is centre
            coord_names=["lat", "lon"]
        )
    <xarray.DataArray (lat: 36, lon: 72)> Size: 21kB
    array([[nan, nan, nan, ..., nan, nan, nan],
           [nan, nan, nan, ..., nan, nan, nan],
           [nan, nan, nan, ..., nan, nan, nan],
           ...,
           [nan, nan, nan, ..., nan, nan, nan],
           [nan, nan, nan, ..., nan, nan, nan],
           [nan, nan, nan, ..., nan, nan, nan]], shape=(36, 72))
    Coordinates:
      * lat      (lat) float64 288B -87.5 -82.5 -77.5 ... 77.5 82.5 87.5
      * lon      (lon) float64 576B -177.5 -172.5 ... 172.5 177.5
    """
    if not isinstance(resolution, Iterable):
        resolution = [resolution for _ in range(len(bounds))]
    if len(resolution) != len(coord_names) or len(bounds) != len(coord_names):
        raise ValueError("Input lists must have the same length")
    coords = {
        c_name: np.arange(lbound, ubound, res)
        for c_name, (lbound, ubound), res in zip(
            coord_names, bounds, resolution
        )
    }
    grid = xr.DataArray(coords=xr.Coordinates(coords))
    return grid


def assign_to_grid(
    values: np.ndarray,
    grid_idx: np.ndarray,
    grid: xr.DataArray,
    fill_value: Any = np.nan,
) -> xr.DataArray:
    """
    Assign a vector of values to a grid, using a list of grid index values.

    Parameters
    ----------
    values : numpy.ndarray
        The values to map onto the output grid.
    grid_idx : numpy.ndarray
        The 1d index of the grid (assuming "C" style ravelling) for each value.
    grid : xarray.DataArray
        The grid used to define the output grid.
    fill_value : Any
        The value to fill unassigned grid boxes. Must be a valid value of the
        input `values` data type.

    Returns
    -------
    out_grid : xarray.DataArray
        A new grid containing the values mapped onto the grid.
    """
    values = values.reshape(-1)
    grid_idx = grid_idx.reshape(-1)

    # Check that the fill_value is valid
    values_dtype = values.dtype
    fill_value_dtype = type(fill_value)
    if not np.can_cast(fill_value_dtype, values_dtype):
        raise TypeError(
            f"Type of input 'fill_value' ({fill_value}: {fill_value_dtype}) "
            + f"is not valid for values data type: {values_dtype}."
        )

    out_grid = xr.DataArray(
        data=np.full(grid.shape, fill_value=fill_value, dtype=values_dtype),
        coords=grid.coords,
    )
    coords_to_assign = np.unravel_index(grid_idx, out_grid.shape, "C")
    out_grid.values[coords_to_assign] = values

    return out_grid


def grid_to_distance_matrix(
    grid: xr.DataArray,
    dist_func: Callable = haversine_distance_from_frame,
    lat_coord: str = "lat",
    lon_coord: str = "lon",
    **dist_kwargs,
) -> xr.DataArray:
    """
    Calculate a distance matrix between all positions in a grid. Orientation of
    latitude and longitude will be maintained in the returned distance matrix.

    Parameters
    ----------
    grid : xarray.DataArray
        A 2-d grid containing latitude and longitude indexes specified in
        decimal degrees.
    dist_func : Callable
        Distance function to use to compute pairwise distances. See
        glomar_gridding.distances.calculate_distance_matrix for more
        information.
    lat_coord : str
        Name of the latitude coordinate in the input grid.
    lon_coord : str
        Name of the longitude coordinate in the input grid.
    **dist_kwargs
        Keyword arguments to pass to the distance function.

    Returns
    -------
    dist : xarray.DataArray
        A DataArray containing the distance matrix with coordinate system
        defined with grid cell index ("index_1" and "index_2"). The coordinates
        of the original grid are also kept as coordinates related to each
        index (the coordinate names are suffixed with "_1" or "_2"
        respectively).

    Examples
    --------
    >>> grid = grid_from_resolution(
            resolution=5,
            bounds=[(-87.5, 90), (-177.5, 180)],  # Lower bound is centre
            coord_names=["lat", "lon"]
        )
    >>> grid_to_distance_matrix(grid, lat_coord="lat", lon_coord="lon")
    <xarray.DataArray 'dist' (index_1: 2592, index_2: 2592)> Size: 54MB
    array([[    0.        ,    24.24359308,    48.44112457, ...,
            19463.87158499, 19461.22915012, 19459.64166305],
           [   24.24359308,     0.        ,    24.24359308, ...,
            19467.56390938, 19463.87158499, 19461.22915012],
           [   48.44112457,    24.24359308,     0.        , ...,
            19472.29905588, 19467.56390938, 19463.87158499],
           ...,
           [19463.87158499, 19467.56390938, 19472.29905588, ...,
                0.        ,    24.24359308,    48.44112457],
           [19461.22915012, 19463.87158499, 19467.56390938, ...,
               24.24359308,     0.        ,    24.24359308],
           [19459.64166305, 19461.22915012, 19463.87158499, ...,
               48.44112457,    24.24359308,     0.        ]],
          shape=(2592, 2592))
    Coordinates:
      * index_1  (index_1) int64 21kB 0 1 2 3 4 ... 2587 2588 2589 2590 2591
      * index_2  (index_2) int64 21kB 0 1 2 3 4 ... 2587 2588 2589 2590 2591
        lat_1    (index_1) float64 21kB -87.5 -87.5 -87.5 ... 87.5 87.5
        lon_1    (index_1) float64 21kB -177.5 -172.5 ... 172.5 177.5
        lat_2    (index_2) float64 21kB -87.5 -87.5 -87.5 ... 87.5 87.5 87.5
        lon_2    (index_2) float64 21kB -177.5 -172.5 ... 172.5 177.5
    """
    coords = grid.coords
    out_coords = cross_coords(coords)

    dist: np.ndarray = calculate_distance_matrix(
        pl.DataFrame(
            {
                lat_coord: out_coords[f"{lat_coord}_1"].values,
                lon_coord: out_coords[f"{lon_coord}_1"].values,
            }
        ),
        dist_func=dist_func,
        lat_col=lat_coord,
        lon_col=lon_coord,
        **dist_kwargs,
    )

    return xr.DataArray(
        dist,
        coords=xr.Coordinates(out_coords),
        name="dist",
    )


def cross_coords(
    coords: xr.Coordinates | xr.Dataset | xr.DataArray,
) -> xr.Coordinates:
    """
    Combine a set of coordinates into a cross-product, for example to construct
    a coordinate system for a distance matrix.

    For example a coordinate system defined by:
        lat = [0, 1],
        lon = [4, 5],
    would yield a new coordinate system defined by:
        index_1 = [0, 1, 2, 3]
        index_2 = [0, 1, 2, 3]
        lat_1 = [0, 0, 1, 1]
        lon_1 = [4, 5, 4, 5]
        lat_2 = [0, 0, 1, 1]
        lon_2 = [4, 5, 4, 5]

    Parameters
    ----------
    coords : xarray.Coordinates | xarray.DataArray | xarray.Dataset
        The set of coordinates to combine, or cross. This should be of length
        2 and have names defined by `lat_coord` and `lon_coord` input arguments.
        The ordering of the coordinates will define the cross ordering. If an
        array is provided then the coordinates are extracted.

    Returns
    -------
    cross_coords : xarray.Coordinates
        The new crossed coordinates, including index, and each of the input
        coordinates, for each dimension.

    Examples
    --------
    >>> grid = grid_from_resolution(
            resolution=5,
            bounds=[(-87.5, 90), (-177.5, 180)],  # Lower bound is centre
            coord_names=["lat", "lon"]
        )
    >>> cross_coords(grid.coords)
    Coordinates:
      * index_1  (index_1) int64 21kB 0 1 2 3 4 ... 2587 2588 2589 2590 2591
      * index_2  (index_2) int64 21kB 0 1 2 3 4 ... 2587 2588 2589 2590 2591
        lat_1    (index_1) float64 21kB -87.5 -87.5 -87.5 ... 87.5 87.5
        lon_1    (index_1) float64 21kB -177.5 -172.5 ... 172.5 177.5
        lat_2    (index_2) float64 21kB -87.5 -87.5 -87.5 ... 87.5 87.5 87.5
        lon_2    (index_2) float64 21kB -177.5 -172.5 ... 172.5 177.5
    """
    if isinstance(coords, (xr.DataArray, xr.Dataset)):
        coords = coords.coords
    dims = coords.dims

    coord_df = pl.from_records(
        list(coords.to_index()),
        schema=list(dims),  # type: ignore
        orient="row",
    )

    n = coord_df.height
    cross_coords: dict[str, Any] = {"index_1": range(n), "index_2": range(n)}
    for i in range(1, 3):
        cross_coords.update(
            {f"{c}_{i}": (f"index_{i}", coord_df[c]) for c in coord_df.columns}
        )

    return xr.Coordinates(cross_coords)
