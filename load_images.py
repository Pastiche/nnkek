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

import io

import oss2
import requests
from docopt import docopt
from tools.util.img.io_utils import pil_download_image_by_aeid
from tools.util.img.mode_utils import as_rgb
from datetime import datetime
from nnkek.utils import parallel_processor
import pandas as pd
from multiprocessing import cpu_count

args = docopt(__doc__)

oss_dir = 'aeru_img/'


def img_to_oss(im_id, bucket):
    im = pil_download_image_by_aeid(im_id, verbose=True)
    im = as_rgb(im, always_copy=False)
    buffer = io.BytesIO()

    if im_id.endswith('png'):
        im.save(buffer, format='png')
    else:
        im.save(buffer, format='jpeg', quality=95)

    bucket.put_object(f'{oss_dir}{im_id}', buffer.getvalue())


def url2oss(img_id, bucket):
    url = f'https://ae01.alicdn.com/kf/{img_id}'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36',
        'From': 'ivashkovdv@gmail.com'
    }

    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            print(img_id, r.status_code, r.text)
        else:
            bucket.put_object(f'{oss_dir}{img_id}', r)
            print(f'success {img_id}, {datetime.now().time()}')

    except Exception as e:
        print(e)


if __name__ == '__main__':
    # o = easy_odps.EasyODPS('aer_ds_dev',
    #                        'LTAI4FffSvYoGWwnDcvc62im',
    #                        'S05Zj2RDPL4Xm51Tlhlhlh5AQBDohv')
    # o.save_query2csv('SELECT image_name FROM aeru_img;',
    #                  'data/alibaba/csv/img_ru_20201015.csv')

    auth = oss2.AnonymousAuth()
    endpoint = 'http://oss-cn-hangzhou-zmf.aliyuncs.com'
    bucket = oss2.Bucket(auth, endpoint, 'bucketaeru')

    df = pd.read_csv('data/alibaba/csv/img_ru_20201015.csv')

    df.dropna(inplace=True)
    img_ids = df.image_name.values

    print(f'All: {len(img_ids)}')

    stored = []

    for obj in oss2.ObjectIterator(bucket, prefix=oss_dir, delimiter='/'):
        if not obj.is_prefix():  # folder
            stored.append(obj.key.split('/')[1])

    stored_set = set(stored)
    assert len(stored) == len(stored_set)
    print(f'stored: {len(stored)}')

    n_cpu = cpu_count()

    img_ids = set(img_ids) - stored_set

    parallel_processor(list(img_ids),
                       url2oss,
                       bucket=bucket,
                       n_jobs=n_cpu*2)