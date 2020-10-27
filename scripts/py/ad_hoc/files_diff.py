import os.path as osp
from glob import glob

predicts = [osp.basename(x).split("_")[0] for x in glob("predict/ru/*")]
clusters = [osp.basename(x).split("_")[0] for x in glob("tables/clusters/ru/*")]
diff = set(predicts) - set(clusters)
print(diff)
print(len(diff))
