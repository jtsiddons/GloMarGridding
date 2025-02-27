"""Internal module containing functions used by NOC"""

from warnings import warn
import polars as pl
import os
import glob
import xarray as xr

from glomar_gridding.utils import check_cols


def _get_clim_qc(var: str) -> list[str]:
    match var.lower():
        case "at":
            return [
                "noval_at",
                "hardlim_at",
                "nonorm_at",
                "clim_at",
            ]
        case "sst":
            return [
                "noval_sst",
                "hardlim_sst",
                "nonorm_sst",
                "clim_sst",
                "freez_sst",
            ]
        case "slp":
            return [
                "noval_slp",
                "clim_slp",
                "nonorm_slp",
            ]
        case "dpt":
            return [
                "noval_dpt",
                "clim_dpt",
                "nonorm_dpt",
                "ssat",
            ]
        case _:
            warn(f"No climatological qc columns for {var}")
            return []


def add_height_adjustment(
    df: pl.DataFrame,
    height_adjustment_path: str,
    year: int,
    adjusted_height: int,
    height_member: int,
    mat_col: str = "obs_anomalies",
) -> pl.DataFrame:
    """
    Adjust MAT height measurement using a height adjustment file

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing UIDS and MAT values to adjust
    height_adjustment_path : str
        Path to the height adjustment file
    year : int
        Year of the data (used as part of the filename)
    height_member : int
        Height adjustment ensemble member to use (defines column to load). If
        the height_member is 0 then no adjustment is applied.
    adjusted_height : int
        Height to adjust to (used as part of the filename). One of 2, 10, 20
    mat_col : str
        Name of the MAT column to adjust

    Returns
    -------
    joined_df : pandas.DataFrame
        With height adjustment applied to the MAT column
    """
    if adjusted_height not in [2, 10, 20]:
        raise ValueError(f"Adjustment height {adjusted_height} is not valid")
    if height_member == 0:
        return df

    hadj_file = f"MAT_hgt_{year}_t{adjusted_height}m.feather"
    hadj_file = os.path.join(height_adjustment_path, hadj_file)
    if not os.path.isfile(hadj_file):
        raise FileNotFoundError(
            f"Height adjustment file: {hadj_file} not found."
        )
    hadj_col = f"end.{height_member}"
    df = df.join(
        pl.read_ipc(hadj_file, columns=["uid", hadj_col], memory_map=False),
        on="uid",
        how="inner",
    )
    df = df.with_columns(
        (pl.col(mat_col) - pl.col(hadj_col)).alias(mat_col)
    ).drop(hadj_col)
    return df


def merge_ellipse_params(
    ellipse_monthly_array: xr.Dataset,
    df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Join an xarray.Dataset containing NOC ellipse parameters to an observation
    dataframe.

    Parameters
    ----------
    ellipse_monthly_array : xarray.Dataset
        The Dataset containing the ellipse parameters, which are the following
        parameters defining the ellipse at a given gridcell lat/lon:
            - "lx": the length of the major axis
            - "ly": the length of the minor axis
            - "theta": the angle of the ellipse positive such that a value of
              0 would represent an ellipse with the major axis aligned
              east-west.
    df : polars.DataFrame
        The observational dataframe, containing "grid_lat" and
        "grid_lon" columns used to join the ellipse parameters.

    Returns
    -------
    df : polars.DataFrame
        With additional columns "grid_lx", "grid_lx", "grid_theta"
        containing the ellipse parameters for a given gridcell.
    """
    required_cols = ["grid_lat", "grid_lon"]
    check_cols(df, required_cols)
    ellipse_df = pl.from_pandas(
        ellipse_monthly_array.to_dataframe().reset_index(drop=False)
    )

    df = df.join(
        ellipse_df,
        left_on=["grid_lat", "grid_lon"],
        right_on=["latitude", "longitude"],
    )
    df = df.rename({"lx": "grid_lx", "ly": "grid_ly", "theta": "grid_theta"})
    return df


def read_groups(
    path: str,
    schema: dict[str, pl.DataType] | None = None,
    **kwargs,
) -> pl.DataFrame:
    """
    Read a series of feather files identified by a format string and globbing.

    Parameters
    ----------
    path : str
        Path to the files. Can include named format blocks and globbing.
    **kwargs
        Keywords indicating replacements for the format strings
    """
    if kwargs:
        path = path.format(**kwargs)
    files: list[str] = glob.glob(path)
    if not files:
        warn(f"No files found from {path = }")
        if schema:
            return pl.DataFrame(schema=schema)
        return pl.DataFrame()

    if not schema:

        def _reader(file) -> pl.DataFrame:
            return pl.read_ipc(file, memory_map=False)
    else:

        def _reader(file) -> pl.DataFrame:
            return pl.read_ipc(
                file,
                memory_map=False,
                columns=list(schema.keys()),
            ).cast(schema)

    return pl.concat(map(_reader, files), how="diagonal")


def load_icoads(
    path: str,
    qc_path: str,
    var: str,
    year: int,
    month: int,
) -> pl.DataFrame:
    """
    Load the ICOADS data from output of PyCOADS for the given year and month.

    Parameters
    ----------
    path : str
        Path to the main input data, which includes the climatological columns.
        The value `path` can contain format replacements for "year" and "month",
        the values for which are replaced with the year and month arguments.
    qc_path : str
        Path to the tracking and duplicate qc flags. Can contain formats
        for "year" and "month".
    var : str
        Name of the variable to extract data for. Also used to identify
        climatological qc columns.
    year : int
        Year of the data. Used as the replacement value in the format strings
        for path, qc_path.
    month : int
        Month of the data. Used as the replacement value in the format strings
        for path, qc_path.

    Returns
    -------
    df : polars.DataFrame
        The ICOADS data plus climatological, duplicate, and track QC flags.
    """
    clim_qc_cols = _get_clim_qc(var)
    data_columns = {
        "yr": pl.UInt16,
        "mo": pl.UInt8,
        "dy": pl.UInt8,
        "hr": pl.Float32,
        "lat": pl.Float32,
        "lon": pl.Float32,
        var: pl.Float32,
        "ii": pl.UInt8,
        "id": pl.String,
        "uid": pl.String,
        "dck": pl.UInt16,
    }
    data_columns = data_columns.update({k: pl.Boolean for k in clim_qc_cols})
    qc_columns = {
        "uid": pl.String,
        "dck": pl.UInt16,
        "datetime": pl.Datetime,
        "local_datetime": pl.Datetime,
        "orig_id": pl.String,
        "data_type": pl.String,
        "any_flag": pl.Boolean,
        "point_dup_flag": pl.UInt8,
        "track_dup_flag": pl.Boolean,
    }

    df = read_groups(path, data_columns, year=year, month=month)
    qc_df = read_groups(qc_path, qc_columns, year=year, month=month)

    if qc_df.height == 0:
        raise ValueError("No data, or don't have the flags")
    else:
        qc_df = qc_df.filter(~pl.any_horizontal(qc_columns))
        # print(qc_df)

        # when merging with FINAL_PROC datafiles, which are not pre-appended
        qc_df = qc_df.with_columns(pl.col("uid").str.slice(-6).name.keep())

        # extra bit to check for duplicates in uid
        if qc_df["uid"].is_duplicated().any():
            raise ValueError("Data contains duplicated UIDs.")
        # duplicate_values = qc_df["uid"].is_duplicated()
        # print(duplicate_values[duplicate_values== True])
        # remove duplicate values in uid column
        qc_df.drop("dck", strict=True)
        qc_df = qc_df.filter(
            ~pl.col("any_flag")
            & pl.col("point_dup_flag").le(1)
            & ~pl.col("track_dup_flag")
        )
        # print('QC DF', qc_df) #.columns)

    return df.join(qc_df, on="uid", how="inner").sort("datetime")


def get_git_commit() -> str:
    """
    Get the most recent commit, used for debugging/traceability of noc gridding
    runs.

    Raises an error if the `.git/logs/HEAD` file cannot be found
    """
    git_path = os.path.join(os.path.dirname(__file__), "..", ".git")
    # Handle worktrees
    if os.path.isfile(git_path):
        with open(git_path, "r") as io:
            git_path = io.readline().split()[1]

    if not os.path.isdir(git_path):
        raise FileNotFoundError("Cannot locate '.git' directory")

    git_log_path = os.path.join(git_path, "logs", "HEAD")
    if not os.path.isfile(git_log_path):
        raise FileNotFoundError("Cannot locate git log file")

    with open(git_log_path, "r") as io:
        last_line = io.readlines()[-1]
    git_commit = last_line.split()[0]
    return git_commit
