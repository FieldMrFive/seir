import numpy as np
import cv2 as cv
import os
from mxnet.gluon.data import Dataset

__all__ = ["RasterImageDataset"]


class RasterImageDataset(Dataset):
    def __init__(self, config):
        super(RasterImageDataset, self).__init__()

        self._data_dir = config["dataset"]["data_dir"]
        self._data_lst = []
        with open(os.path.join(self._data_dir, config["dataset"]["data_lst"])) as fin:
            for line in fin:
                self._data_lst.append(line.strip('\n'))
        self._length = len(self._data_lst)

    def __getitem__(self, idx):
        data_name = self._data_lst[idx]
        img = cv.imread(os.path.join(self._data_dir, data_name + ".jpg"), cv.IMREAD_COLOR)
        img = np.transpose(img, (2, 0, 1))
        label = []
        with open(os.path.join(self._data_dir, data_name + ".txt"), "r") as fin:
            for line in fin:
                label += [float(i) for i in line.strip('\n').split(' ')]
        return np.array(img, dtype=np.float32), np.array(label, dtype=np.float32)

    def __len__(self):
        return self._length
