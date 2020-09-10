"""run img load

Usage:
    run_model_fit [options]
    run_model_fit --help

Options:
  -h --help                     Show this screen.
  -d --img_dir <d>              Path where to dump the images.
                                [default: /data/alibaba/img]
  -u --urls <u>                 Path to txt file with img urls separated by \n.
                                [default: /data/alibaba/img]
"""
from pathlib import Path

import pandas as pd
from nnkek.imagers import ImgLoader
from docopt import docopt

args = docopt(__doc__)


def load_img(img_name):
    return imloader.download_to_folder(image_aeid=img_name,
                                       file_name=img_name,
                                       return_path=True)


if __name__ == '__main__':

    #TODO: Completely reimplement according to bestpractices and script you gave
    # to Alexander

    img_dir = Path(args['--img_dir']).resolve()
    imloader = ImgLoader(img_dir)

    # load links
    df = pd.read_csv('data/alibaba/csv/items24k_20200727.csv')

    # get first (main) image name only
    df["main_im"] = df.img_path.apply(lambda x: x.split(',')[0])
    df.dropna(inplace=True)

    assert df.shape[0] == 24534

    # load images to disk
    df["img_path"] = df["img_name"].apply(load_img)
    df.dropna(inplace=True)

    df["img_id"] = df.img_path.apply(lambda x: x.split('.')[0])

    df.to_csv('data/alibaba/csv/top_items_ru.csv', index=False)