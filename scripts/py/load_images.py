"""run img load

Usage:
    load_images [options]
    load_images --help

Options:
  -h --help                     Show this screen.
  -d --img_dir <d>              Path where to dump the images.
                                [default: /var/alibaba/img]
  -u --urls <u>                 Path to txt file with img urls separated by \\n.
                                [default: /var/alibaba/txt/img.txt]
  -w --overwrite                Overwrite existing images
  -v --verbose                  Print exceptions and results
  -o --offset <o>               From which image to start [default: 0]
  -c --cpu <c>                  Number of cpu to use [default: -1]
                                (-1 means mp.cpu_count()*2)
"""

import multiprocessing as mp
import os
from time import sleep

import requests
from PIL import Image
from docopt import docopt

from nnkek.utils import parallel_processor

args = docopt(__doc__)


def pil_download_image_by_url(url: str, file_name: str, overwrite=False, verbose=False):

    if not overwrite and os.path.exists(file_name):
        if verbose:
            print(f"{file_name} exists, skipping..")
        return
    try:
        r = requests.get(
            url,
            stream=True,
            timeout=20,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.105 Safari/537.36",
                "From": "hackerman@dark.net",
            },
        )
        if r.status_code == 200:
            r.raw.decode_content = True
            img = Image.open(r.raw)
            with open(file_name, "wb") as f:
                img.save(f)
                if verbose:
                    print(f"{file_name} success!")
    except Exception as e:
        if verbose:
            print(e)


def pil_download_image_by_aeid(image_aeid: str, dump_dir: str, overwrite=False, verbose=False):
    file_name = os.path.join(dump_dir, image_aeid)
    return pil_download_image_by_url(
        f"https://ae01.alicdn.com/kf/{image_aeid}",
        file_name=file_name,
        overwrite=overwrite,
        verbose=verbose,
    )


if __name__ == "__main__":
    print(args)

    with open(args["--urls"], "r") as f:
        img_ids = list(map(lambda x: x.strip(), f.readlines()))

    img_ids = img_ids[int(args["--offset"]) :]
    print(f"All: {len(img_ids)}")

    n_cpu = int(args["--cpu"]) if int(args["--cpu"]) > 0 else mp.cpu_count() * 2
    print(f"{n_cpu} cpu is used..")

    sleep(3)

    parallel_processor(
        img_ids,
        worker=pil_download_image_by_aeid,
        verbose=args["--verbose"],
        dump_dir=args["--img_dir"],
        overwrite=args["--overwrite"],
    )
