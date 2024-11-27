import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.figure import Figure
from typing import Any, Dict, Iterable, Optional


def projected_scatter(fig: Figure, ax: GeoAxes,
                      lons: Iterable, lats: Iterable,
                      add_colorbar: bool = False,
                      title: Optional[str] = None,
                      skwargs: Dict[str, Any] = {},
                      ckwargs: Dict[str, Any] = {},
                      land_col: Optional[str] = None,
                      ocean_col: Optional[str] = None,
                      ) -> None:
    """
        `projected_scatter`

    Function to plot a simple scatter on a projected axis.

    Args
    ----

    fig : matplotlib.figure.Figure
        The Figure object - used for adding colorbar
    ax : cartopy.mpl.geoaxes.GeoAxes
        The axis on which to add the plot. If transform is not a key in
        skwargs then the transformation is determined from the axis. If
        that value is 'cartopy.crs.Robinson' then 'cartopy.crs.PlateCarree' is
        used.
        Ref: https://scitools.org.uk/cartopy/docs/latest/gallery/scalar_data/wrapping_global.html#sphx-glr-gallery-scalar-data-wrapping-global-py
    lons : Iterable
        The longitudinal positions to plot
    lats : Iterable
        The latitudinal positions to plot
    add_colorbar : bool
        Optionally add colorbar to the plot
    title : str | None
        Title of the plot
    skwargs : dict[str, any]
        Keyword arguments to pass to ax.scatter. This can include the transform
        or the variable to colour the points by for example.
    ckwargs : dict[str, any]
        Keyword agruments to pass to fig.colorbar. Can include the colour map
        for example.
    land_col : str | None
        Colour for the land, land is not included if set to None
    ocean_col : str | None
        Colour for the ocean, ocean is not included if set to None

    Return
    ------

    The input fig, ax with the scatter and optional colorbar included
    """
    lonextent = [-180., 180.] #min(lons), max(lons)]
    latextent = [-90., 90.] #[min(lats), max(lats)]
    extent = lonextent + latextent

    if 'transform' not in skwargs:
        proj = ax.projection
        if proj == ccrs.Robinson():
            proj = ccrs.PlateCarree()
        skwargs['transform'] = proj
    ax.set_extent(extent, crs=skwargs['transform'])

    pcm = ax.scatter(lons, lats, **skwargs)
    if add_colorbar:
        fig.colorbar(pcm, ax=ax, **ckwargs)
        
    if land_col is not None:
        ax.add_feature(cfeature.LAND, color=land_col)
    if ocean_col is not None:
        ax.add_feature(cfeature.OCEAN, color=ocean_col)
    ax.coastlines() 

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True) #, dms=True, x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False
    if title is not None:
        ax.set_title(title, pad=20)
    #plt.show()
    return None
