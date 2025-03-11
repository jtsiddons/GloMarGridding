"""Tests of the distances module"""

import pytest  # noqa: F401
from math import sqrt
import polars as pl
from glomar_gridding.distances import euclidean_distance, haversine_distance


def test_euclidean():
    R = 6371.0
    df = pl.DataFrame({"lat": [-90, 90, 0], "lon": [0, 0, 23]})

    dist = euclidean_distance(df, radius=R)
    print(dist)

    assert dist[0, 0] == dist[1, 1] == dist[2, 2] == 0.0
    assert dist[0, 1] == pytest.approx(2 * R)
    assert dist[0, 2] == pytest.approx(sqrt(2) * R)


def test_haversine():
    R = 6371.0
    halifax = (44.6476, -63.5728)
    southampton = (50.9105, -1.4049)
    expected = 4557  # From Google

    df = pl.from_records(
        [halifax, southampton], orient="row", schema=["lat", "lon"]
    )

    dist = haversine_distance(df, radius=R)

    assert dist[0, 0] == dist[1, 1] == 0.0
    assert dist[0, 1] == pytest.approx(expected, abs=1)  # Allow 1km out
