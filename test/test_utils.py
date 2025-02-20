import pytest
from glomar_gridding.io import get_recurse


def test_nested_dict() -> None:
    test_dict = {
        "nested": {"a": 4, "nested_2": {"a": 6, "b": 3}},
        "a": 2,
        "b": 9
        }

    assert get_recurse(test_dict, "c") is None
    assert get_recurse(test_dict, "a") == 2
    assert get_recurse(test_dict, "nested", "a") == 4
    assert get_recurse(test_dict, "nested", "b") is None
    assert get_recurse(test_dict, "nested", "b", default="DEFAULT") == "DEFAULT"
    assert get_recurse(test_dict, "nested", "nested_2", "a") == 6
    return None
