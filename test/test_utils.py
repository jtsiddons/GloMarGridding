import pytest  # noqa: F401
import numpy as np
import polars as pl
from datetime import date

from glomar_gridding.io import get_recurse
from glomar_gridding.grid import grid_from_resolution
from glomar_gridding.utils import (
    select_bounds,
    filter_bounds,
    get_pentad_range,
    days_since_by_month,
    batched,
)


def test_nested_dict() -> None:
    test_dict = {
        "nested": {"a": 4, "nested_2": {"a": 6, "b": 3}},
        "a": 2,
        "b": 9,
    }

    assert get_recurse(test_dict, "c") is None
    assert get_recurse(test_dict, "a") == 2
    assert get_recurse(test_dict, "nested", "a") == 4
    assert get_recurse(test_dict, "nested", "b") is None
    assert get_recurse(test_dict, "nested", "b", default="DEFAULT") == "DEFAULT"
    assert get_recurse(test_dict, "nested", "nested_2", "a") == 6
    return None


def test_bounds() -> None:
    var_names = ["latitude", "longitude"]
    bounds = [(-71.0, -23.0), (127.0, 149.0)]

    grid = grid_from_resolution(
        resolution=5,
        bounds=[(-87.5, 90), (-177.5, 180)],
        coord_names=var_names,
    )

    result = select_bounds(
        x=grid,
        bounds=bounds,
        variables=var_names,
    )

    n = 10_000
    lons = 180 - 360 * np.random.rand(n)
    lats = 90 - 180 * np.random.rand(n)
    df = pl.DataFrame({"longitude": lons, "latitude": lats})

    df = df.pipe(
        filter_bounds,
        bounds=bounds,
        bound_cols=var_names,
        closed="none",
    )

    assert np.max(result["latitude"].values) < -23
    assert np.min(result["latitude"].values) > -71
    assert np.max(result["longitude"].values) < 149
    assert np.min(result["longitude"].values) > 127

    print(df.height)
    assert df.height > 0
    assert df.get_column("latitude").max() < -22  # type: ignore
    assert df.get_column("latitude").min() > -71  # type: ignore
    assert df.get_column("longitude").max() < 149  # type: ignore
    assert df.get_column("longitude").min() > 127  # type: ignore


@pytest.mark.parametrize(
    "name, centre, start, end",
    [
        ("generic", date(1966, 9, 22), date(1966, 9, 20), date(1966, 9, 24)),
        ("leap", date(2004, 2, 28), date(2004, 2, 26), date(2004, 3, 2)),
        ("leap_day", date(2004, 2, 29), date(2004, 2, 27), date(2004, 3, 2)),
        ("no leap", date(1999, 2, 28), date(1999, 2, 26), date(1999, 3, 2)),
    ],
)
def test_pentads(name, centre, start, end) -> None:
    pentad_start, pentad_end = get_pentad_range(centre)

    assert pentad_start == start
    assert pentad_end == end


def test_batched():
    assert list(batched("ABCDEFG", 3)) == [
        ("A", "B", "C"),
        ("D", "E", "F"),
        ("G",),
    ]
    assert list(batched("ABCDEFG", 2)) == [
        ("A", "B"),
        ("C", "D"),
        ("E", "F"),
        ("G",),
    ]


@pytest.mark.parametrize(
    "name, year, day",
    [
        ("generic", 1999, 15),
        ("leap", 2004, 15),
        ("leap_day", 2004, 29),
    ],
)
def test_days(name, year, day):
    dates = [date(year, m, day) for m in range(1, 13)]
    diffs = [(d - date(year, 1, day)).total_seconds() // 86400 for d in dates]

    calc_diff = days_since_by_month(year, day)

    assert all(d == c for d, c in zip(diffs, calc_diff))
