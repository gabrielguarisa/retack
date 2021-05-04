from pydoc import locate
from typing import Any, Dict, List, Tuple, Union


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


def load_elements(data: Dict[str, Any]) -> Dict[str, List[Any]]:
    output = {"elements": [], "names": [], "args": []}
    for info in data:
        element, name, args = load_single_element(info)
        output["elements"].append(element)
        output["names"].append(unique_name(name, output["names"]))
        output["args"].append(args)

    return output
