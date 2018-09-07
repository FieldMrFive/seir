#!/usr/bin/env python

import argparse
import logging
import os
import mxnet as mx
import numpy as np
import cv2 as cv

from seir.utils import *
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

    eval_dataset = RasterImageDataset(data_dir=params_config["dataset"]["data_dir"],
                                      data_lst_file=params_config["dataset"]["train_data_lst"])
    resolution = params_config["dataset"]["resolution"]
    horizon = params_config["dataset"]["horizon"]

    net = mx.gluon.nn.SymbolBlock.imports(symbol_file=prefix + "-symbol.json",
                                          input_names=["data0", "data1"],
                                          param_file=prefix + "-%04d.params" % checkpoint)

    eval_metric = ["mse", AlongTrackError(name="ate"), CrossTrackError(name="cte"), Displacement(name="dpm")]
    if not isinstance(eval_metric, mx.metric.EvalMetric):
        eval_metric = mx.metric.create(eval_metric)

    eval_data = mx.gluon.data.DataLoader(dataset=eval_dataset, batch_size=1)
    for i, (data, state, label) in enumerate(eval_data):
        pred = net(data, state)
        eval_metric.update(labels=label, preds=pred)
        img = np.array(data[0].asnumpy()[2] * 60, dtype=np.uint8)
        img += (data[0].asnumpy()[0] * 30).astype(np.uint8)

        pred = pred.asnumpy()
        label = label.asnumpy()
        for j in range(horizon):
            pixel = (data.shape[3] / 2 - int(pred[0, j * 2 + 1] / resolution),
                     data.shape[2] / 2 - int(pred[0, j * 2] / resolution))
            cv.circle(img, pixel, 6, (255), -1)

        for j in range(horizon):
            pixel = (data.shape[3] / 2 - int(label[0, j * 2 + 1] / resolution),
                     data.shape[2] / 2 - int(label[0, j * 2] / resolution))
            cv.circle(img, pixel, 6, (100))

        name_value = eval_metric.get_name_value()
        for j, (name, value) in enumerate(name_value):
            cv.putText(img, "%s=%.4f" % (name, value), (50, 50 * j), cv.FONT_HERSHEY_COMPLEX, 1, (200, 100, 120))
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
