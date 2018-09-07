import numpy as np
import mxnet as mx
import os
from mxnet.gluon.data import Dataset

__all__ = ["RasterTensorDataset"]


class RasterTensorDataset(Dataset):
    def __init__(self, data_dir, data_lst_file):
        super(RasterTensorDataset, self).__init__()

        self._data_dir = data_dir
        self._data_lst = []
        with open(os.path.join(self._data_dir, data_lst_file)) as fin:
            for line in fin:
                self._data_lst.append(line.strip('\n'))
        self._length = len(self._data_lst)

    def __getitem__(self, idx):
        data_name = self._data_lst[idx]
        [img] = mx.nd.load(os.path.join(self._data_dir, data_name + ".data"))
        label = []
        state = []
        with open(os.path.join(self._data_dir, data_name + ".label"), "r") as fin:
            for line in fin:
                label += [float(i) for i in line.strip('\n').split(' ')]
        with open(os.path.join(self._data_dir, data_name + ".state"), "r") as fin:
            for line in fin:
                state.append(float(line.strip('\n')))
        return img, mx.nd.array(state), mx.nd.array(label)

    def __len__(self):
        return self._length