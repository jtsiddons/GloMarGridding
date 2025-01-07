"""Functions for creating grids and mapping observations to a grid"""

from collections.abc import Iterable
from dataclasses import dataclass
import numpy as np
import pandas as pd
import xarray as xr

from .utils import find_nearest, select_bounds, select_bounds_3d


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


# TODO: Ensure indices map correctly (row or column major ordering)
# TODO: Detect row/column major ordering from the DataArray and use the same
def align_to_grid(
    obs: pd.DataFrame,
    grid: xr.DataArray,
    grid_latname: str = "latitude",
    grid_lonname: str = "longitude",
    obs_latname: str = "lat",
    obs_lonname: str = "lon",
    bounds: GridBounds | None = None,
) -> pd.DataFrame:
    """
    Align an observation dataframe to a grid defined by an xarray DataArray.

    Maps observations to the nearest grid-point, and sorts the data by the
    1d index of the DataArray in a row-major format.

    The grid defined by the latitude and longitude coordinates of the input
    DataArray is then used as the output grid of the Gridding process.

    Parameters
    ----------
    obs : pandas.DataFrame
        The observational DataFrame containing positional data with latitude,
        longitude values within the `obs_latname` and `obs_lonname` columns
        respectively. Observations are mapped to the nearest grid-point in the
        grid.
    grid : xarray.DataArray
        Contains the grid coordinates to map observations to.
    grid_latname : str
        Name of the latitude coordinate in the input grid DataArray.
    grid_lonname : str
        Name of the longitude coordinate in the input grid DataArray.
    obs_latname : str
        Name of the column containing latitude values in the input
        observational DataFrame.
    obs_lonname : str
        Name of the column containing longitude values in the input
        observational DataFrame.
    bounds : GridBounds | None
        An optional object of type `GridBounds` which contains boundary values
        in the `west`, `east`, `south`, and `north` attributes. This will also
        be used to filter the observations (observations outside of the bounds
        will be excluded from the interpolation).

    Returns
    -------
    obs : pandas.DataFrame
        Containing additional `grid_lon`, `grid_lat`, and `grid_idx` values
        indicating the longitude, latitude, and grid index of the observation
        respectively. The DataFrame is also sorted (ascendingly) by the
        `grid_idx` columns for consistency with the gridding functions.
    """
    if bounds is not None:
        grid = select_bounds(
            grid,
            (bounds.west, bounds.east),
            (bounds.south, bounds.north),
            grid_lonname,
            grid_latname,
        )
        obs = obs.loc[
            (bounds.west <= obs[obs_lonname] < bounds.east)
            & (bounds.south <= obs[obs_latname] < bounds.north)
        ]

    grid_lats = grid.coords[grid_latname].values
    grid_lons = grid.coords[grid_lonname].values
    grid_size = grid.shape

    grid_lat_idx, obs_to_grid_lat = find_nearest(grid_lats, obs[obs_latname])
    grid_lon_idx, obs_to_grid_lon = find_nearest(grid_lons, obs[obs_lonname])

    flattened_idx = np.ravel_multi_index(
        ([grid_lat_idx, grid_lon_idx]),
        grid_size,
        order="C",  # row-major
    )

    obs["grid_lat"] = obs_to_grid_lat
    obs["grid_lon"] = obs_to_grid_lon
    obs["grid_idx"] = flattened_idx
    obs.sort_values(by="grid_idx", inplace=True, ascending=True)

    return obs


# TODO: Ensure indices map correctly (row or column major ordering)
# TODO: Detect row/column major ordering from the DataArray and use the same
def align_to_grid_3d(
    obs: pd.DataFrame,
    grid: xr.DataArray,
    grid_latname: str = "latitude",
    grid_lonname: str = "longitude",
    grid_depthname: str = "depth",
    obs_latname: str = "lat",
    obs_lonname: str = "lon",
    obs_depthname: str = "depth",
    bounds: GridBounds3d | None = None,
) -> pd.DataFrame:
    """
    Align an observation dataframe to a 3d grid defined by an xarray DataArray.

    Maps observations to the nearest grid-point, and sorts the data by the
    1d index of the DataArray in a row-major format.

    The grid defined by the latitude and longitude coordinates of the input
    DataArray is then used as the output grid of the Gridding process.

    Parameters
    ----------
    obs : pandas.DataFrame
        The observational DataFrame containing positional data with latitude,
        longitude values within the `obs_latname` and `obs_lonname` columns
        respectively. Observations are mapped to the nearest grid-point in the
        grid.
    grid : xarray.DataArray
        Contains the grid coordinates to map observations to.
    grid_latname : str
        Name of the latitude coordinate in the input grid DataArray.
    grid_lonname : str
        Name of the longitude coordinate in the input grid DataArray.
    grid_depthname : str
        Name of the depth coordinate in the input grid DataArray.
    obs_latname : str
        Name of the column containing latitude values in the input observational
        DataFrame.
    obs_lonname : str
        Name of the column containing longitude values in the input
        observational DataFrame.
    obs_depthname : str
        Name of the column containing depth values in the input observational
        DataFrame.
    bounds : GridBounds3d | None
        An optional object of type `GridBounds3d` which contains boundary values
        in the `west`, `east`, `south`, `north`, `bottom`, and `top` attributes.
        This will also be used to filter the observations (observations outside
        of the bounds will be excluded from the interpolation).

    Returns
    -------
    obs : pandas.DataFrame
        Containing additional `grid_lon`, `grid_lat`, and `grid_idx` values
        indicating the longitude, latitude, and grid index of the observation
        respectively. The DataFrame is also sorted (ascendingly) by the
        `grid_idx` columns for consistency with the gridding functions.
    """
    if bounds is not None:
        grid = select_bounds_3d(
            grid,
            (bounds.west, bounds.east),
            (bounds.south, bounds.north),
            (bounds.bottom, bounds.top),
            grid_lonname,
            grid_latname,
            grid_depthname,
        )
        obs = obs.loc[
            (bounds.west <= obs[obs_lonname] < bounds.east)
            & (bounds.south <= obs[obs_latname] < bounds.north)
            & (bounds.bottom <= obs[obs_depthname] < bounds.top)
        ]

    grid_lats = grid.coords[grid_latname].values
    grid_lons = grid.coords[grid_lonname].values
    grid_depths = grid.coords[grid_depthname].values
    grid_size = grid.shape

    grid_lat_idx, obs_to_grid_lat = find_nearest(grid_lats, obs[obs_latname])
    grid_lon_idx, obs_to_grid_lon = find_nearest(grid_lons, obs[obs_lonname])
    grid_depth_idx, obs_to_grid_depth = find_nearest(
        grid_depths, obs[obs_depthname]
    )

    flattened_idx = np.ravel_multi_index(
        ([grid_lat_idx, grid_lon_idx, grid_depth_idx]),
        grid_size,
        order="C",  # row-major
    )

    obs["grid_lat"] = obs_to_grid_lat
    obs["grid_lon"] = obs_to_grid_lon
    obs["grid_depth"] = obs_to_grid_depth
    obs["grid_idx"] = flattened_idx
    obs.sort_values(by="grid_idx", inplace=True, ascending=True)

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
