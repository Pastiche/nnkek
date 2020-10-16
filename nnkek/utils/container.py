from functools import reduce
from typing import Dict, Any, Mapping


# container traversing
def get_by_path(container: Mapping[str, Any], path: str, default: Any = None) -> Any:
    """
    Retrieves a value from the container using a specified path.
    If the path goes through a list, the first element is taken.
    :param container: initial mapping container
    :param path: nested path with a full stop separator
    :param default: default value to return if path is invalid or no value found
    :return a value found in the container using path or default value
    """

    def func(container: Dict[str, Any], key):
        if not container:
            return None

        if isinstance(container, Mapping):
            return container.get(key)

        if not isinstance(container, list):
            return None

        if isinstance(container[0], Mapping):
            return container[0].get(key)

        return None

    value = reduce(func, path.split("."), container)
    return value if value is not None else default


def get_list_column(dictionary, list_path, column):
    collection = get_by_path(dictionary, list_path)

    if not collection:
        return None

    if not isinstance(collection, list):
        return None

    if not isinstance(collection[0], dict):
        return None

    return [get_by_path(element, column) for element in collection]


def get_list_element_field(container: Mapping[str, Any], list_path: str, field: str, element_index: int) -> Any:
    """
    Returns a value of the selected list element field. The list is acquired
    from the container using a specified path.
    :param container: a mapping container for list to search
    :param list_path: path to a list with a full stop separator
    :param field: the name of the field which value to return
    :param element_index: index of the list element which's field value to
    return
    """
    list_column = get_list_column(container, list_path, field)

    if not list_column:
        return None

    if len(list_column) < element_index + 1:
        return None

    return list_column[element_index]
