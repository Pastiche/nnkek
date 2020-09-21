"""run img load

Usage:
    run_img_load [options]
    run_img_load --help

Options:
  -h --help                     Show this screen.
  -d --img_dir <d>              Path where to dump the images.
                                [default: var/alibaba/img]
  -u --urls <u>                 Path to txt file with img urls separated by \n.
                                [default: var/alibaba/txt/img.txt]
  -w --overwrite                Overwrite existing images
  -v --verbose                  Print exceptions and results
  -o --offset <o>               From which image to start [default: 0]
  -c --cpu <c>                  Number of cpu to use [default: -1]
                                (-1 means mp.cpu_count()*2)
"""

import multiprocessing as mp
import os
from functools import partial
from typing import Callable, Any, Sequence
from time import sleep

import requests
from PIL import Image
from docopt import docopt

args = docopt(__doc__)


def pil_download_image_by_url(
        url: str,
        file_name: str,
        overwrite=False,
        verbose=False
):

    if not overwrite and os.path.exists(file_name):
        if verbose:
            print(f'{file_name} exists, skipping..')
        return
    try:
        r = requests.get(url, stream=True, timeout=20, headers={
            'User-Agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.105 Safari/537.36",
            'From': 'hackerman@dark.net'
        })
        if r.status_code == 200:
            r.raw.decode_content = True
            img = Image.open(r.raw)
            with open(file_name, 'wb') as f:
                img.save(f)
                if verbose:
                    print(f'{file_name} success!')
    except Exception as e:
        if verbose:
            print(e)


def pil_download_image_by_aeid(
        image_aeid: str,
        dump_dir: str,
        overwrite=False,
        verbose=False
):
    file_name = os.path.join(dump_dir, image_aeid)
    return pil_download_image_by_url(
        f"https://ae01.alicdn.com/kf/{image_aeid}",
        file_name=file_name,
        overwrite=overwrite,
        verbose=verbose
    )


def parallel_processor(sequence: Sequence[Any], worker: Callable, n_jobs=-1,
                       **worker_kwargs) -> Sequence[Any]:
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
        raise ValueError(f'Got invalid n_jobs argument: {n_jobs}')

    if n_jobs == 1:
        return worker(sequence, **worker_kwargs)

    with mp.Pool(processes=n_jobs) as pool:
        results = pool.map(partial(worker, **worker_kwargs), sequence)

    return results


def increase_value(value):
    with value.get_lock():
        value.value += 1
        return value.value


if __name__ == '__main__':
    print(args)

    with open(args['--urls'], 'r') as f:
        img_ids = list(map(lambda x: x.strip(), f.readlines()))

    img_ids = img_ids[int(args['--offset']):]
    print(f'All: {len(img_ids)}')

    n_cpu = int(args['--cpu']) if int(args['--cpu']) > 0 else mp.cpu_count()*2
    print(f'{n_cpu} cpu is used..')

    sleep(3)

    parallel_processor(
        img_ids,
        worker=pil_download_image_by_aeid,
        verbose=args['--verbose'],
        dump_dir=args['--img_dir'],
        overwrite=args['--overwrite']
    )
