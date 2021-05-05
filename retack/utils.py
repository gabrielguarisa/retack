import inspect
from pydoc import locate
from typing import Any, Dict, List, Tuple, Union

import numpy as np


def import_element(import_str: str) -> object:
    elem = locate(import_str)
    if elem is None:
        raise ImportError(f"Failed to import {import_str}")
    return elem


def unique_name(original_name: str, list_of_names: List[str]) -> str:
    i = 1
    tmp_name = original_name
    while tmp_name in list_of_names:
        tmp_name = f"{original_name}_{i}"
        i += 1
    return tmp_name


def load_single_element(
    data: Union[Dict[str, Any], str]
) -> Tuple[type, str, Dict[str, Any]]:
    element = None
    element_name = None
    element_args = {}
    if isinstance(data, dict):
        import_str = list(data.keys())[0]
        element = import_element(import_str)
        element_name = data[import_str].get("name", element.__name__)
        element_args = data[import_str].get("args", {})
    elif isinstance(data, str):
        element = import_element(data)
        element_name = element.__name__
    else:
        raise TypeError("Invalid YAML format!")

    return element, element_name, element_args


def load_elements(data: List[Any]) -> Dict[str, List[Any]]:
    output = {"elements": [], "names": [], "args": []}
    for info in data:
        element, name, args = load_single_element(info)
        output["elements"].append(element)
        output["names"].append(unique_name(name, output["names"]))
        output["args"].append(args)

    return output


def set_params_to_dict(
    params: Union[Dict[str, Any], List[Any]], variable_name: str = None
):
    if len(params) == 0:
        raise ValueError(
            "The number of {} must be greater than zero!".format(
                "params" if variable_name is None else variable_name
            )
        )
    output = {}
    if isinstance(params, list) or isinstance(params, np.ndarray):
        for i in range(len(params)):
            name = unique_name(
                params[i].__name__
                if inspect.isfunction(params[i])
                else params[i].__class__.__name__,
                list(output.keys()),
            )
            output[name] = params[i]
    elif isinstance(params, dict):
        output = params
    else:
        if variable_name is None:
            raise TypeError("Invalid type!")
        raise TypeError(f"Invalid type for {variable_name}!")
    return output
