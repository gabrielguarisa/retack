from pydoc import locate
from typing import List


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
