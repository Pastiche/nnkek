import asyncio
import multiprocessing as mp
from functools import partial
from functools import reduce
from tools.util import file_utils as futils
from typing import Dict, Callable, Any, Sequence, Mapping
import numpy as np
import os.path as osp

from tqdm import tqdm


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


# parallel mapping
async def map_io(sequence: Sequence[Any], worker: Callable, **worker_kwargs) -> Sequence[Any]:
    """
    Асинхронно (IO) маппит последовательность.
    :param sequence: последовательность для обработки
    :param worker: IO функция-обработчик, должна принимать первым параметром
    элемент для обработки
    :param worker_kwargs: аргументы вызова обработчика
    :return: преобразованная последовательность. Каждый элемент - результат
    обработки или возникшее исключение
    """
    tasks = [worker(elem, **worker_kwargs) for elem in sequence]
    return await asyncio.gather(*tasks, return_exceptions=True)


async def map_io_iter(sequence: Sequence[Any], worker: Callable, batch_size=50, **worker_kwargs) -> Sequence[Any]:
    """
    Асинхронно (IO) итеративно (батчами) маппит последовательность
    :param sequence: последовательность для обработки
    :param worker: IO функция-обработчик, должна принимать первым параметром
    элемент для обработки
    :param batch_size: размер батча единовременно вызываемых IO операций
    :param worker_kwargs: аргументы вызова обработчика
    :return: преобразованная последовательность. Каждый элемент - результат
    обработки или возникшее исключение
    """
    res = []
    with tqdm() as pbar:
        for i in range(0, len(sequence), batch_size):
            batch = sequence[i : i + batch_size]

            responses_batch = await map_io(batch, worker, **worker_kwargs)
            res.extend(responses_batch)

            pbar.update()

    return res


def batch_parallel(
    sequence: Sequence[Any], worker: Callable, n_jobs=-1, batch_size=50, **worker_kwargs
) -> Sequence[Any]:
    """
    Мультипоточно итеративно (батчами) маппит последовательность
    :param n_jobs: число используемых процессов (по-умолчанию, все доступные)
    :param sequence: последовательность для обработки
    :param worker: функция-обработчик, должна принимать первым параметром
    элемент для обработки
    :param batch_size: размер батча единовременно вызываемых операций
    :param worker_kwargs: аргументы вызова обработчика
    :return: преобразованная последовательность. Каждый элемент - результат
    обработки или возникшее исключение
    """
    res = []
    with tqdm() as pbar:
        for i in range(0, len(sequence), batch_size):
            batch = sequence[i : i + batch_size]

            responses_batch = parallel_processor(batch, worker, n_jobs, **worker_kwargs)
            res.extend(responses_batch)

            pbar.update()

    return res


def parallel_processor(sequence: Sequence[Any], worker: Callable, n_jobs=-1, **worker_kwargs) -> Sequence[Any]:
    """
    Параллельный преобразователь последовательности.

    :param sequence: последовательность
    :param worker: функция-обработчик последовательности
    :param worker_kwargs: аргументы вызова обработчика
    :param n_jobs: число используемых процессов (по-умолчанию, все доступные)
    :return: преобразованная последовательность
    """
    if n_jobs == -1:
        n_jobs = mp.cpu_count() - 1

    if not isinstance(n_jobs, int) or n_jobs < 1:
        raise ValueError(f"Got invalid n_jobs argument: {n_jobs}")

    if n_jobs == 1:
        return worker(sequence, **worker_kwargs)

    with mp.Pool(processes=n_jobs) as pool:
        results = pool.map(partial(worker, **worker_kwargs), sequence)

    return results


# paths manipulations
def map_img_paths(image_folder: str, img_names: Sequence[str] = None) -> dict:
    """Given files root folder and their unique names returns mapping to corresponding paths from the given root"""
    all_paths = {osp.basename(x): x for x in futils.path_get_file_list(image_folder, ["image"])}
    return {x: all_paths.get(x) for x in img_names} if img_names else all_paths


def get_img_paths(image_folder: str, img_names: Sequence[str] = None) -> np.array:
    """Given files root folder and their unique names returns corresponding paths from the given root"""
    img_names2paths = map_img_paths(image_folder, img_names)
    return np.array([x for x in img_names2paths.values()])
