import pandas as pd
import pytest
from sklearn.svm import SVC

from retack.utils import (
    get_element_name,
    get_instance_or_class,
    import_element,
    load_element,
    unique_name,
)


def test_import_element():
    elem = import_element("pandas.DataFrame")
    assert elem is pd.DataFrame


def test_import_invalid_element():
    with pytest.raises(ImportError):
        _ = import_element("pandas.dataframe")


@pytest.mark.parametrize(
    "original_name,list_of_names,expected_name",
    [
        ("name", ["name", "test"], "name_1"),
        ("name", ["test"], "name"),
        ("test", ["test", "test", "test_1", "test_a"], "test_2"),
        ("tmp", [], "tmp"),
    ],
)
def test_unique_name(original_name, list_of_names, expected_name):
    assert unique_name(original_name, list_of_names) == expected_name


@pytest.mark.parametrize(
    "data,expected_element,expected_element_name,expected_element_args",
    [
        ("pandas.DataFrame", pd.DataFrame, "DataFrame", {}),
        (
            {
                "pandas.DataFrame": {
                    "name": "Batatinha",
                    "args": {"criterion": "poisson"},
                }
            },
            pd.DataFrame,
            "Batatinha",
            {"criterion": "poisson"},
        ),
        (
            {"pandas.DataFrame": {"args": {"criterion": "poisson"}}},
            pd.DataFrame,
            "DataFrame",
            {"criterion": "poisson"},
        ),
        (
            {"pandas.DataFrame": {"name": "Batatinha"}},
            pd.DataFrame,
            "Batatinha",
            {},
        ),
    ],
)
def test_load_element(
    data, expected_element, expected_element_name, expected_element_args
):
    info = load_element(data)
    assert info["element"] is expected_element
    assert info["name"] == expected_element_name
    assert info["args"] == expected_element_args


def test_load_invalid_single_element():
    with pytest.raises(TypeError):
        _ = load_element(["pandas.DataFrame"])


@pytest.mark.parametrize(
    "element,expected",
    [
        (SVC, "SVC"),
        (SVC(), "SVC"),
        (unique_name, "unique_name"),
    ],
)
def test_get_element_name(element, expected):
    assert get_element_name(element) == expected


@pytest.mark.parametrize(
    "element,return_instance,expected",
    [
        (SVC, True, SVC),
        (SVC, False, SVC),
        (SVC(), True, SVC),
        (SVC(), False, SVC),
    ],
)
def test_get_instance_or_class(element, return_instance, expected):
    result = get_instance_or_class(
        element=element, return_instance=return_instance
    )
    if return_instance:
        assert isinstance(result, expected)
    else:
        assert result == expected
