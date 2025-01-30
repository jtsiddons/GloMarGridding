import polars as pl
import os


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
