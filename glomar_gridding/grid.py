"""Functions for creating grids and mapping observations to a grid"""

from collections.abc import Iterable
from dataclasses import dataclass
import numpy as np
import polars as pl
import xarray as xr

from .utils import filter_bounds, find_nearest, select_bounds


@dataclass
class GridBounds:
    """Class for simple grid bounds"""

    west: float
    east: float
    south: float
    north: float


@dataclass
class GridBounds3d(GridBounds):
    """Class for simple grid bounds"""

    west: float
    east: float
    south: float
    north: float
    bottom: float
    top: float


def align_to_grid(
    obs: pl.DataFrame,
    grid: xr.DataArray,
    grid_coords: list[str] = ["latitude", "longitude"],
    obs_coords: list[str] = ["lat", "lon"],
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
    grid_coords : list[str]
        Names of the coordinates in the input grid DataArray used to define the
        grid.
    obs_coords : list[str]
        Names of the column containing positional values in the input
        observational DataFrame.
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

    obs[grid_prefix + "idx"] = flattened_idx
    if add_grid_pts:
        for grid_pos, obs_coord in zip(obs_to_grid_pos, obs_coords):
            obs[grid_prefix + obs_coord] = grid_pos

    if sort:
        obs = obs.sort("grid_idx", descending=False)

    return obs


def grid_from_resolution(
    resolution: float | list[float],
    bounds: GridBounds,
    coord_names: tuple[str, str],
) -> xr.DataArray:
    if not isinstance(resolution, Iterable):
        resolution = [resolution, resolution]
    return _grid_from_resolution_2d(resolution, bounds, coord_names)


def grid_from_resolution_3d(
    resolution: float | list[float],
    bounds: GridBounds3d,
    coord_names: tuple[str, str, str],
) -> xr.DataArray:
    if not isinstance(resolution, Iterable):
        resolution = [resolution, resolution, resolution]
    return _grid_from_resolution_3d(resolution, bounds, coord_names)


def grid_from_pos(
    lats: np.ndarray,
    lons: np.ndarray,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
) -> xr.DataArray:
    coords = xr.Coordinates({lat_name: lats, lon_name: lons})
    return xr.DataArray(coords=coords)


def grid_from_pos_3d(
    lats: np.ndarray,
    lons: np.ndarray,
    depths: np.ndarray,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    depth_name: str = "depth",
) -> xr.DataArray:
    coords = xr.Coordinates(
        {lat_name: lats, lon_name: lons, depth_name: depths}
    )
    return xr.DataArray(coords=coords)


def _grid_from_resolution_2d(
    resolution: list[float],
    bounds: GridBounds,
    coord_names: tuple[str, str],
) -> xr.DataArray:
    lat_name, lon_name = coord_names

    lat_res, lon_res, *_ = resolution
    lat_coords = np.arange(bounds.west, bounds.east, lat_res)
    lon_coords = np.arange(bounds.south, bounds.north, lon_res)
    return grid_from_pos(lat_coords, lon_coords, lat_name, lon_name)


def _grid_from_resolution_3d(
    resolution: list[float],
    bounds: GridBounds3d,
    coord_names: tuple[str, str, str],
) -> xr.DataArray:
    lat_name, lon_name, depth_name = coord_names

    lat_res, lon_res, depth_res, *_ = resolution
    lat_coords = np.arange(bounds.west, bounds.east, lat_res)
    lon_coords = np.arange(bounds.south, bounds.north, lon_res)
    depth_coords = np.arange(bounds.bottom, bounds.top, depth_res)
    return grid_from_pos_3d(
        lat_coords, lon_coords, depth_coords, lat_name, lon_name, depth_name
    )
