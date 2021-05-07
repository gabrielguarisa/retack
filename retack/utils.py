import inspect
from pydoc import locate
from typing import Any, Callable, Dict, List, Tuple, Type, Union


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


def load_element(
    data: Union[Dict[str, Any], str]
) -> Tuple[type, str, Dict[str, Any]]:
    output = {}
    if isinstance(data, dict):
        import_str = list(data.keys())[0]
        output["element"] = import_element(import_str)
        output["name"] = data[import_str].get(
            "name", output["element"].__name__
        )
        output["args"] = data[import_str].get("args", {})
    elif isinstance(data, str):
        output["element"] = import_element(data)
        output["name"] = output["element"].__name__
        output["args"] = {}
    else:
        raise TypeError("Invalid YAML format!")

    return output


def get_element_name(element: Union[Callable, Type, Any]) -> str:
    if inspect.isfunction(element) or inspect.isclass(element):
        return element.__name__
    try:
        return element.__class__.__name__
    except Exception:  # pragma: no cover
        raise TypeError("Invalid element name!")


def get_instance_or_class(element: Any, return_instance: bool = True) -> Any:
    if return_instance:
        return element() if inspect.isclass(element) else element
    return element if inspect.isclass(element) else element.__class__
