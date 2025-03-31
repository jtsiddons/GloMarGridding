"""
Requires numpy, scipy, sklearm
iris needs to be installed (it is required by other modules within this package
xarray cubes should work via iris interface)
"""

import math
from typing import Any, Literal

import iris
import iris.coords as icoords
import iris.util as iutil
import numpy as np
from numpy import linalg, ma
from sklearn.metrics.pairwise import euclidean_distances, haversine_distances

# from astropy.constants import R_earth
from glomar_gridding.constants import (
    DEFAULT_N_JOBS,
    RADIUS_OF_EARTH_KM,
)
from glomar_gridding.ellipse import (
    MODEL_TYPE,
    MODEL_TYPE_TO_SUPERCATEGORY,
    SUPERCATEGORY_PARAMS,
    MaternEllipseModel,
)


DeltaXMethod = Literal["Met_Office", "Modified_Met_Office"]


class CovarianceCube:
    """
    Class to build spatial covariance and correlation matricies
    Interacts with MLE_c_ij_Builder_Karspeck to build covariance models

    Parameters
    ----------
    data_cube : instance of iris.cube.Cube
        Training data stored within iris cube

        Some modification probably needed to work with instances
        of xa.dataarray
        The biggest hurdle is that iris cube handle masked data differently
        than xarray

        While both iris.cube.Cube.data and xa.DataArray.data are instances
        to numpy.ndarray...

        A masked array under iris.cube.Cube.data is always an instance of
        np.ma.MaskedArray
        In xarray, xa.DataArray.data is always unmasked.
    """

    def __init__(
        self,
        data_cube,
    ) -> None:
        # Defining the input data
        self.data_cube = data_cube
        self.xy_shape = self.data_cube[0].shape
        if len(self.xy_shape) != 2:
            raise ValueError(
                "Time slice maps should be 2D; check extra dims (ensemble?)"
            )
        self.big_covar_size = np.prod(self.xy_shape)

        self._parse_coords()
        self._detect_mask()

        # Calculate the actual covariance and correlation matrix:
        self._calc_cov(correlation=True)

        return None

    def _parse_coords(self) -> None:
        # Check input data_cube is actually usable
        self.tcoord_pos = -1
        self.xycoords_pos = []
        self.xycoords_name = []
        for i, coord in enumerate(self.data_cube.coords()):
            if coord.standard_name == "time":
                self.tcoord_pos = i
            if coord.standard_name in ["latitude", "longitude"]:
                self.xycoords_pos.append(i)
                self.xycoords_name.append(coord.standard_name)
        if self.tcoord_pos == -1:
            raise ValueError("Input cube needs a time dimension")
        if self.tcoord_pos != 0:
            raise ValueError("Input cube time dimension not at 0")
        if len(self.xycoords_pos) != 2:
            raise ValueError(
                "Input cube need two spatial dimensions "
                + "('latitude' and 'longitude')"
            )
        self.xycoords_pos = tuple(self.xycoords_pos)
        # if 'lat' in self.xycoords_name[0]:
        #     self.xycoords_name = self.xycoords_name[::-1]

        # Length of time dimension
        self.time_n = len(self.data_cube.coord("time").points)

        return None

    def _detect_mask(self) -> None:
        # Detect data mask and determine dimension of array without masked data
        # Almost certain True near the coast
        self.data_has_mask = ma.is_masked(self.data_cube.data)
        if self.data_has_mask:
            # Depending on dataset, the mask might not be invariant
            # (like sea ice)
            # xys with time varying mask are currently discarded.
            # If analysis is conducted seasonally
            # this should normally not a problem unless in high latitudes
            self.cube_mask = np.any(self.data_cube.data.mask, axis=0)
            self.cube_mask_1D = self.cube_mask.flatten()
            self._self_mask()
            self.small_covar_size = np.sum(np.logical_not(self.cube_mask))
        else:
            self.cube_mask = np.zeros_like(
                self.data_cube[0].data.data, dtype=bool
            )
            self.cube_mask_1D = self.cube_mask.flatten()
            self.small_covar_size = self.big_covar_size

        # Look-up table of the coordinates
        self.xx, self.yy = np.meshgrid(
            self.data_cube.coord("longitude").points,
            self.data_cube.coord("latitude").points,
        )
        self.xm = ma.masked_where(self.cube_mask, self.xx)
        self.ym = ma.masked_where(self.cube_mask, self.yy)
        self.xy = np.column_stack([self.xm.compressed(), self.ym.compressed()])
        self.xy_full = np.column_stack([self.xm.flatten(), self.ym.flatten()])

        return None

    def _calc_cov(
        self,
        correlation: bool = False,
        rounding: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        correlation : bool
            to state if you want a correlation (normalised covariance or not)
        rounding : int
            round the values of the output
        """
        # Reshape data to (t, xy),
        # get rid of mask values -- cannot caculate cov for such data
        xyflatten_data = self.data_cube.data.reshape(
            (
                len(self.data_cube.coord("time").points),
                self.big_covar_size,
            )
        )
        xyflatten_data = ma.compress_rowcols(xyflatten_data, -1)
        # Remove mean --
        # even data that says "SST anomalies" don't have zero mean (?!)
        xy_mean = np.mean(xyflatten_data, axis=0, keepdims=True)
        xyflatten_data = xyflatten_data - xy_mean
        self.cov = np.matmul(np.transpose(xyflatten_data), xyflatten_data)
        # xy_cov = _AT_A(xyflatten_data)
        if correlation:
            return self._cov2cor(rounding=rounding)
        if rounding is not None:
            self.cov = np.round(self.cov, rounding)

    def _cov2cor(
        self,
        rounding: int | None = None,
    ) -> None:
        """
        Normalises the covariance matrices within the class instance
        and return correlation matrices
        https://gist.github.com/wiso/ce2a9919ded228838703c1c7c7dad13b

        Parameters
        ----------
        rounding : int
            round the values of the output
        """
        stdevs = np.sqrt(np.diag(self.cov))
        normalisation = np.outer(stdevs, stdevs)
        self.cor = self.cov / normalisation
        self.cor[self.cov == 0] = 0
        # sigma_inverse = np.zeros_like(self.Cov)
        # np.fill_diagonal(sigma_inverse,
        #                  np.reciprocal(np.sqrt(np.diagonal(self.Cov))))
        # ans = np.matmul(np.matmul(sigma_inverse, self.Cov), sigma_inverse)
        # #ans = _Dinv_A_Dinv(self.Cov)
        if rounding is not None:
            self.cor = np.round(self.cor, rounding)
        return None

    def calc_distance_angle_selfcube(
        self,
        haversine: bool = False,
        compressed: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute distances between grid points of cube data stored
        within class instance

        Parameters
        ----------
        haversine : bool
            Use the haversine instead
        compressed : bool
            Do a np.ma.MaskedArray.comressed, this get rids of masked points...

        Returns
        -------
        D, A : np.ndarray [float]
            D (distance in km), A (distance in angle)
        """
        xx, yy = np.meshgrid(
            self.data_cube.coord("longitude").points,
            self.data_cube.coord("latitude").points,
        )
        xx = ma.masked_where(self.cube_mask, xx)
        yy = ma.masked_where(self.cube_mask, yy)
        if compressed:
            unmeshed_x = xx.compressed()
            unmeshed_y = yy.compressed()
        else:
            unmeshed_x = xx.flatten()
            unmeshed_y = yy.flatten()
        D, A = self._calc_distance_angle(
            unmeshed_x, unmeshed_y, haversine=haversine
        )
        return D, A

    def _calc_distance_angle(
        self,
        unmeshed_x: np.ndarray,
        unmeshed_y: np.ndarray,
        haversine: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute distance and angle between all points."""
        yx_og = np.column_stack([unmeshed_y, unmeshed_x])
        if haversine:
            # Great circle - Earth
            yx = np.deg2rad(yx_og)
            D = haversine_distances(yx) * RADIUS_OF_EARTH_KM
        else:
            # Delta degrees - locally flat Earth, treating like it an image
            yx = yx_og
            D = euclidean_distances(yx)

        # Angle difference vs x/longitude-axis - below two yields the same
        # result
        # _dy = euclidean_distances(
        #     np.column_stack([unmeshed_y, np.zeros_like(unmeshed_y)])
        # )
        # _dx = euclidean_distances(
        #     np.column_stack([np.zeros_like(unmeshed_x), unmeshed_x])
        # )
        _dy = np.abs(np.subtract.outer(yx[:, 0], yx[:, 0]))
        dy = np.triu(_dy) - np.tril(_dy)

        _dx = np.abs(np.subtract.outer(yx[:, 1], yx[:, 1]))
        dx = np.triu(_dx) - np.tril(_dx)
        # In radians; note, arctan2 can tell if dy and dx are negative
        A = np.arctan2(dy, dx)
        ##
        return D, A  # Both D and A are square matrices

    def _self_mask(self) -> None:
        """Broadcast cube_mask to all observations"""
        broadcasted_mask = np.broadcast_to(
            self.cube_mask, self.data_cube.data.shape
        )
        self.data_cube.data = ma.masked_where(
            broadcasted_mask, self.data_cube.data
        )

    def _reverse_mask_from_compress_1d(
        self,
        compressed_1D_vector: np.ndarray,
        fill_value: float = 0.0,
        dtype=np.float32,
    ) -> np.ndarray:
        """
        Since there are lot of flatten and compressing going on for observations
        and fitted parameters, this reverses the 1D array to the original 2D map

        DANGER WARNING, use different fill_value depending on situation
        This affects how signal and image processing module interacts
        with missing and masked values.
        They don't ignore them, so a fill_value like 0 may be sensible
        for covariance (-999.99 will do funny things if it finds its way to a
        convolution and filter) iris doesn't care, but should use something like
        -999.99 or something.

        Parameters
        ----------
        compressed_1D_vector : np.ndarray (1D) shape = (NUM,)
            1D vector that is a compressed/flatten version of a 2D map
        fill_value: float
            Fill value for masked point
        dtype: valid numpy float type
            The dtype for 2D array.

        Returns
        -------
        ans : np.ndarray (2D) shape = (NUM,NUM)
            The 2D map array
        """
        compressed_counter = 0
        ans = np.zeros_like(self.cube_mask_1D, dtype=dtype)
        for i in range(len(self.cube_mask_1D)):
            if not self.cube_mask_1D[i]:
                ans[i] = compressed_1D_vector[compressed_counter]
                compressed_counter += 1
        ma.set_fill_value(ans, fill_value)
        ans = ma.masked_where(self.cube_mask_1D, ans)
        return ans

    def remap_one_point_2_map(
        self,
        compressed_vector: np.ndarray,
        cube_name: str = "stuff",
        cube_unit: str = "1",
    ) -> iris.cube.Cube:
        """
        Reverse one row/column of the covariance/correlation matrix
        to a plottable iris cube using mask defined in class
        """
        dummy_cube = self.data_cube[0, :, :].copy()
        masked_vector = self._reverse_mask_from_compress_1d(compressed_vector)
        dummy_cube.data = masked_vector.reshape(self.xy_shape)
        dummy_cube.rename(cube_name)
        dummy_cube.units = cube_unit
        return dummy_cube

    def fit_ellipse_model(
        self,
        xy_point: int,
        matern_ellipse: MaternEllipseModel,
        max_distance: float = 20.0,
        min_distance: float = 0.3,
        delta_x_method: DeltaXMethod | None = "Modified_Met_Office",
        guesses: list[float] | None = None,
        bounds: list[tuple[float, ...]] | None = None,
        opt_method: str = "Nelder-Mead",
        tol: float = 0.001,
        estimate_SE: str | None = None,
        n_jobs: int = DEFAULT_N_JOBS,
    ) -> dict[str, Any]:
        """
        Fit ellipses/covariance models using adhoc local covariances

        the form of the covariance model depends on the "fform" attribute of the
        Ellipse model:
            isotropic (radial distance only)
            anistropic (x and y are different, but not rotated)
            anistropic_rotated (rotated)

        If the "fform" attribued ends with _pd then phyiscal distances are used
        instead of degrees

        range is defined max_distance (either in km and degrees)
        default is in degrees, but needs to be km if fform is from _pd series
        <--- likely to be wrong: max_distance should only be in degrees

        there is also a min_distance in which values,
        matern function is not defined at the origin, so the 0.0 needs to
        removed

        v = matern covariance function shape parameter
        Karspeck et al and Paciorek and Schervish use 3 and 4
        but 0.5 and 1.5 are popular
        0.5 gives an exponential decay
        lim v-->inf, Gaussian shape

        delta_x_method: only meaningful for _pd fits
            "Spherical_COS_Law": uses COS(C) = COS(A)COS(B)
            "Met_Office": Cylindrical Earth delta_x = 6400km x delta_lon
            (in radians)
            "Modified_Met_Office": uses the average zonal dist at different lat

        Parameters
        ----------
        xy_point : int
            The index point where ellipses will be fitted to

        max_distance : float
            Maximum seperation in distance unit that data will be fed
            into parameter fitting
            Units depend on fform (it is usually either degrees or km)

        min_distance: float
            Minimum seperation in distance unit that data
            will be fed into parameter fitting
            Units depend on fform (it is usually either degrees or km)
            Note: Due to the way we compute the Matern function,
            it is undefined at dist == 0 even if the limit -> zero is obvious.

        delta_x_method="Modified_Met_Office": str
            How to compute distances between grid points
            For istropic variogram/covariances, this is a trival problem;
            you can just take the haversine or
            Euclidian ("tunnel") distance as they are non-directional.

            But it is non trival for anistropic cases,
            you have to define a set of orthogonal space. In HadSST4,
            Earth is assumed to be cyclindrical "tin can" Earth,
            so you can just define the orthogonal space by
            lines of constant lat and lon (delta_x_method="Met_Office").

            The modified "Modified_Met_Office" is a variation to that,
            but allow the tin can get squished at the poles.
            (Sinusoidal projection). This does results in a problem:
            the zonal displacement now depends in which latitude
            you compute on (at the beginning latitude or at the end latitude).
            Here we take the average of the two.

        guesses=None: tuple of floats; None uses default guess values
            Initial guess values that get feeds in the optimizer for MLE.
            In scipy, you are required to do so (but R often doesn't).
            You should anyway; sometimes they do funny things
            if you don't (per recommendation of David Stephenson)

        bounds=None: tuple of floats; None uses default bounds values
            This is essentially a Bayesian "uniformative prior"
            that forces convergence if the optimizer hits the bound.
            For lower resolution fitting, this is rarely a problem.
            For higher resolution fits, this often interacts with
            the limit of the data you can put into the fit the optimizer
            may fail to converge if the input data is very smooth (aka ENSO
            region, where anomalies are smooth over very large (~10000km)
            scales).

        opt_method='Nelder-Mead': str
            scipy optimizer method. Nelder-Mead is the one used by Karspeck.
            See https://docs.scipy.org/doc/scipy/tutorial/optimize.html
            for valid options

        tol=0.001: float
            set convergence tolerance for scipy optimize.
            See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize

            Note on new tol kwarg:
            For N-M, this sets the value to both xatol and fatol
            Default is 1E-4 (?)
            Since it affects accuracy of all values including rotation
            rotation angle 0.001 rad ~ 0.05 deg

        estimate_SE=None : str | None
            The code can estimate the standard error if the Matern parameters.
            This is not usually used or discussed for the purpose of kriging.
            Certain opt_method (gradient descent) can do this automatically
            using Fisher Info for certain covariance funciton,
            but is not possible for some nasty functions (aka Bessel
            func) gets involved nor it is possible for some optimisers
            (such as Nelder-Mead).
            The code does it using bootstrapping.

        n_jobs=DEFAULT_N_JOBS: int
            If parallel processing, number of threads to use.

        Returns
        -------
        ans : dictionary
            Dictionary with results of the fit
            and the observed correlation matrix
        """
        R2 = self.data_cube[0].copy()
        R2x = self._reverse_mask_from_compress_1d(self.cor[xy_point, :])
        R2.data = R2x.reshape(self.xy_shape)
        R2.units = "1"

        X_train, y_train = self._get_train_data(
            xy_point=xy_point,
            min_distance=min_distance,
            max_distance=max_distance,
            anisotropic=matern_ellipse.anisotropic,
            physical_distance=matern_ellipse.physical_distance,
            delta_x_method=delta_x_method,
        )

        results, _, bounds = matern_ellipse.fit(
            X_train,
            y_train,
            guesses=guesses,
            bounds=bounds,
            opt_method=opt_method,
            tol=tol,
            estimate_SE=estimate_SE,
            n_jobs=n_jobs,
        )

        model_params = results.x.tolist()
        stdev = None
        if not matern_ellipse.unit_sigma:
            model_params = results.x.tolist()[:-1]
            stdev = results.x.tolist()[-1]

        # Meaning of fit_success (int)
        # 0: success
        # 1: success but with one parameter reaching lower bounadries
        # 2: success but with one parameter reaching upper bounadries
        # 3: success with multiple parameters reaching the boundaries
        #    (aka both Lx and Ly), can be both at lower or upper boundaries
        # 9: fail, probably due to running out of maxiter
        #    (see scipy.optimize.minimize kwargs "options)"
        if results.success:
            fit_success = _get_fit_score(model_params, bounds, results.nit)
            print("RMSE of multivariate norm fit = ", stdev)
        else:
            print("Convergence fail after ", results.nit, " iterations.")
            print(model_params)
            fit_success = 9
        print("QC flag = ", fit_success)
        # append standard deviation
        model_params.append(np.sqrt(self.cov[xy_point, xy_point] / self.time_n))
        model_params.append(fit_success)
        model_params.append(results.nit)

        v_coord = make_v_aux_coord(matern_ellipse.v)
        template_cube = self._make_template_cube(xy_point)
        model_as_cubelist = create_output_cubes(
            template_cube,
            model_type=matern_ellipse.model_type,
            additional_meta_aux_coords=[v_coord],
            default_values=model_params,
        )["param_cubelist"]

        return {
            "Correlation": R2,
            "MaternObj": matern_ellipse,
            "Model": results,
            "Model_Type": matern_ellipse.model_type,
            "Model_as_1D_cube": model_as_cubelist,
        }

    def _get_train_data(
        self,
        xy_point: int,
        min_distance: float,
        max_distance: float,
        anisotropic: bool,
        physical_distance: bool,
        delta_x_method: DeltaXMethod | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        lonlat = self.xy[xy_point]
        y = self.cor[xy_point, :]
        # dx and dy are in degrees
        dx = np.array([a - lonlat[0] for a in self.xy[:, 0]])
        dy = np.array([a - lonlat[1] for a in self.xy[:, 1]])
        dx[dx > 180.0] -= 360.0
        dx[dx < -180.0] += 360.0
        # distance is in delta-degrees
        distance = linalg.norm(np.column_stack([dx, dy]), axis=1)
        valid_dist_idx = np.where(
            (distance <= max_distance)
            & (distance >= min_distance)
            # Delete the origin (can't have dx = dy = 0)
            & (distance != 0)
        )
        distance = distance[valid_dist_idx]
        dx = dx[valid_dist_idx]
        dy = dy[valid_dist_idx]
        y_train = y[valid_dist_idx]

        if physical_distance:
            if anisotropic:
                distance_jj = np.deg2rad(dy)
                if delta_x_method == "Met_Office":
                    # Cylindrical approximation
                    distance_ii = np.deg2rad(dx)
                elif delta_x_method == "Modified_Met_Office":
                    average_cos = 0.5 * (
                        np.cos(np.deg2rad(lonlat[1] + dy))
                        + np.cos(np.deg2rad(lonlat[1]))
                    )
                    distance_ii = np.deg2rad(dx) * average_cos
                else:
                    raise ValueError(
                        f"Unknown 'delta_x_method': {delta_x_method}"
                    )

                X_train = np.column_stack([distance_ii, distance_jj])
                return X_train * RADIUS_OF_EARTH_KM, y_train
            else:
                latlon = np.array(
                    [np.deg2rad(lonlat[1]), np.deg2rad(lonlat[0])]
                )
                latlon_vector = np.column_stack(
                    [np.deg2rad(lonlat[1] + dy), np.deg2rad(lonlat[0] + dx)]
                )
                latlon = latlon[np.newaxis, :]
                radial = haversine_distances(latlon_vector, latlon)[:, 0]
                return radial.reshape(-1, 1) * RADIUS_OF_EARTH_KM, y_train
        else:
            if anisotropic:
                return np.column_stack([dx, dy]), y_train
            else:
                return distance, y_train

    def find_nearest_xy_index_in_cov_matrix(
        self,
        lonlat: list[float],
        use_full: bool = False,
    ) -> tuple[int, np.ndarray]:
        """
        Find the nearest column/row index of the covariance
        that corresponds to a specific lat lon
        """
        lon, lat, *_ = lonlat

        a = self.xy_full if use_full else self.xy
        idx = int(((a[:, 0] - lon) ** 2.0 + (a[:, 1] - lat) ** 2.0).argmin())
        return idx, a[idx, :]

    def _xy_2_xy_full_index(self, xy_point: int) -> int:
        """
        Given xy index in that corresponding to a latlon
        in the covariance (masked value ma.MaskedArray compressed),
        what is its index with masked values (i.e. ndarray flatten)
        """
        return int(
            np.argwhere(
                np.all((self.xy_full - self.xy[xy_point, :]) == 0, axis=1)
            )[0]
        )

    def _make_template_cube(self, xy_point: int) -> iris.cube.Cube:
        """Make a template cube for lat lon corresponding to xy_point index"""
        xy = self.xy[xy_point, :]
        return self._make_template_cube2(xy)
        # t_len = len(self.data_cube.coord('time').points)
        # template_cube = self.data_cube[t_len // 2].intersection(
        #     longitude=(xy[0] - 0.05, xy[0] + 0.05),
        #     latitude=(xy[1] - 0.05, xy[1] + 0.05),
        # )
        # return template_cube

    def _make_template_cube2(self, lonlat: np.ndarray) -> iris.cube.Cube:
        """Make a template cube for lat lon"""
        t_len = len(self.data_cube.coord("time").points)
        template_cube = self.data_cube[t_len // 2].intersection(
            longitude=(lonlat[0] - 0.05, lonlat[0] + 0.05),
            latitude=(lonlat[1] - 0.05, lonlat[1] + 0.05),
        )
        return template_cube

    def __str__(self):
        return str(self.__class__)
        # return str(self.__class__) + ": " + str(self.__dict__)


def create_output_cubes(
    template_cube,
    model_type: MODEL_TYPE = "ps2006_kks2011_iso",
    default_values: list[float] | None = None,
    additional_meta_aux_coords: list | None = None,
    dtype=np.float32,
) -> dict:
    """
    For data presentation, create template iris cubes to insert data

    Parameters
    ----------
    template_cube : iris.cube.Cube
        A cube to be copied as the template
    model_type : MODEL_TYPE
        See dict at top, string to add auxcoord for outputs
    default_values : list[float]
        Default values to be put in cube, not neccessary a masked/fill value
    additional_meta_aux_coords : list of icoords.AuxCoord
        Whatever additional auxcoord metadata you want to add
    dtype=np.float32 : numpy number type
        The number type to be used in the cube usually np.float32 or
        np.float(64)

    Returns
    -------
    dictionary:
        "param_cubelist"
            Instance of iris.cube.CubeList() that contains cubes to be filled in
        "param_names"
            the names of the variable that get added to param_cubelist
    """
    default_values = default_values or [-999.9, -999.9, -999.9]

    supercategory = MODEL_TYPE_TO_SUPERCATEGORY[model_type]
    params_dict = SUPERCATEGORY_PARAMS[supercategory]

    model_type_coord = icoords.AuxCoord(
        model_type, long_name="fitting_model", units="no_unit"
    )
    supercategory_coord = icoords.AuxCoord(
        supercategory,
        long_name="supercategory_of_fitting_model",
        units="no_unit",
    )

    ans_cubelist = iris.cube.CubeList()
    ans_paramlist = []

    for param, default_value in zip(params_dict, default_values):
        ans_paramlist.append(param)
        param_cube = template_cube.copy()
        param_cube.rename(param)
        param_cube.long_name = param
        param_cube.add_aux_coord(model_type_coord)
        param_cube.add_aux_coord(supercategory_coord)
        param_cube.units = params_dict[param]
        if additional_meta_aux_coords is not None:
            for add_coord in additional_meta_aux_coords:
                param_cube.add_aux_coord(add_coord)
        if param_cube.ndim == 0:
            param_cube = iutil.new_axis(
                iutil.new_axis(param_cube, "longitude"), "latitude"
            )
        param_cube.data[:] = default_value
        param_cube.data = param_cube.data.astype(dtype)
        if ma.isMaskedArray(template_cube.data):
            param_cube.data.mask = template_cube.data.mask
        ans_cubelist.append(param_cube)
    return {"param_cubelist": ans_cubelist, "param_names": ans_paramlist}


def _get_fit_score(model_params, bounds, niter) -> int:
    fit_success: int = 0
    for model_param, bb in zip(model_params, bounds):
        left_check = math.isclose(model_param, bb[0], rel_tol=0.01)
        right_check = math.isclose(model_param, bb[1], rel_tol=0.01)
        left_advisory = "near_left_bnd" if left_check else "not_near_left_bnd"
        right_advisory = "near_right_bnd" if right_check else "not_near_rgt_bnd"
        print(
            "Convergence success after ",
            niter,
            " iterations :) : ",
            model_param,
            bb[0],
            bb[1],
            left_advisory,
            right_advisory,
        )
        if left_check:
            fit_success = 1 if fit_success == 0 else 3
        if right_check:
            fit_success = 2 if fit_success == 0 else 3
    return fit_success


def make_v_aux_coord(v: float) -> icoords.AuxCoord:
    """Create an iris coord for the Matern (positive) shape parameter"""
    return icoords.AuxCoord(v, long_name="v_shape", units="no_unit")
