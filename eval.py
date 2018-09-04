#!/usr/bin/env python

import argparse
import logging
import os
import mxnet as mx
import numpy as np
import cv2 as cv

from seir.utils import load_yaml
from seir.data import RasterImageDataset


def eval(cfg_yaml, checkpoint, log_level):
    params_config = load_yaml(cfg_yaml)

    # set up logs and checkpoint dir
    prefix = os.path.join(params_config["misc"]["checkpoint_dir"], params_config["misc"]["checkpoint_prefix"])

    # set up logger
    log_path = os.path.join(params_config["misc"]["checkpoint_dir"], "eval.log")
    log_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s : %(message)s")

    file_handler = logging.FileHandler(filename=log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_format)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(log_format)

    logger = logging.getLogger("eval")
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    train_data = RasterImageDataset(data_dir=params_config["dataset"]["data_dir"],
                                    data_lst_file=params_config["dataset"]["train_data_lst"])

    net = mx.gluon.nn.SymbolBlock.imports(symbol_file=prefix + "-symbol.json",
                                          input_names=["data0", "data1"],
                                          param_file=prefix + "-%04d.params" % checkpoint)

    for i in range(40, len(train_data)):
        data, state, label = train_data[i]
        net_input = mx.nd.array(data)
        net_input = mx.nd.expand_dims(net_input, axis=0)
        state_input = mx.nd.array(state)
        state_input = mx.nd.expand_dims(state_input, axis=0)
        pred = net(net_input, state_input).asnumpy()[0]

        avg_dist = np.linalg.norm(pred.reshape([-1, 2]) - label.reshape([-1, 2]), axis=1).mean()

        img = np.transpose(data, (1, 2, 0))
        img = np.array(img, dtype=np.uint8)
        img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        for j in range(30):
            pixel = (250 - int(pred[j * 2 + 1] / 0.2), 250 - int(pred[j * 2] / 0.2))
            cv.circle(img, pixel, 6, (75, 255, 150 + j * 22), -1)

        for j in range(30):
            pixel = (250 - int(label[j * 2 + 1] / 0.2), 250 - int(label[j * 2] / 0.2))
            cv.circle(img, pixel, 6, (120, 255, 150 + j * 22))
        img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
        cv.putText(img, "Avg dist: %.3f" % avg_dist, (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (200, 100, 120))
        cv.imshow("test", img)
        cv.waitKey(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=int,
                        help="The checkpoint to eval, e.g 100.")
    parser.add_argument("--cfg", type=str, default="config/train.yaml",
                        help="Path of train config yaml file.")
    parser.add_argument("--log", type=str, default="INFO",
                        help="Log level to console.")
    args = parser.parse_args()
    eval(cfg_yaml=args.cfg,
         checkpoint=args.checkpoint,
         log_level=getattr(logging, args.log.upper(), logging.INFO))
