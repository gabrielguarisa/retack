import pandas as pd
import pytest

from retack.utils import (
    import_element,
    load_single_element,
    set_params_to_dict,
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
def test_load_single_element(
    data, expected_element, expected_element_name, expected_element_args
):
    element, element_name, element_args = load_single_element(data)
    assert element is expected_element
    assert element_name == expected_element_name
    assert element_args == expected_element_args


def test_load_invalid_single_element():
    with pytest.raises(TypeError):
        _ = load_single_element(["pandas.DataFrame"])


def test_set_params_to_dict_without_params():
    with pytest.raises(ValueError):
        _ = set_params_to_dict([])


def test_set_params_to_dict_with_invalid_type():
    with pytest.raises(TypeError):
        _ = set_params_to_dict("pandas.DataFrame")

    with pytest.raises(TypeError):
        _ = set_params_to_dict("pandas.DataFrame", "tmp")


def test_set_params_to_dict():
    from sklearn.metrics import accuracy_score
    from sklearn.svm import SVC

    output = set_params_to_dict(
        [
            SVC(),
            accuracy_score,
            accuracy_score,
        ]
    )
    assert list(output.keys()) == [
        "SVC",
        "accuracy_score",
        "accuracy_score_1",
    ]
