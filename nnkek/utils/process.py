import asyncio
import multiprocessing as mp
from functools import partial
from typing import Callable, Any, Sequence

from tqdm import tqdm


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
