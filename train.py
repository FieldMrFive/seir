#!/usr/bin/env python

import argparse
import logging
import os
import mxnet as mx
from pprint import pformat

from seir.core import Trainer
from seir.data import RasterImageDataset
from seir.models import MobileNetV2
from seir.utils import *


def train(cfg_yaml, gpus, log_level):
    params_config = load_yaml(cfg_yaml)

    # set up logs and checkpoint dir
    checkpoint_dir = params_config["misc"]["checkpoint_dir"]
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # set up logger
    log_path = os.path.join(checkpoint_dir, "train.log")
    log_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s : %(message)s")

    file_handler = logging.FileHandler(filename=log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_format)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(log_format)

    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # print configs
    logger.info("Training setup:\n" + pformat(params_config))

    # init devices
    ctx = [mx.gpu(gpu) for gpu in gpus]

    # load iterator
    train_data = RasterImageDataset(data_dir=params_config["dataset"]["data_dir"],
                                    data_lst_file=params_config["dataset"]["train_data_lst"])
    valid_data = RasterImageDataset(data_dir=params_config["dataset"]["data_dir"],
                                    data_lst_file=params_config["dataset"]["valid_data_lst"])
    multi_gpus_batch_size = params_config["train"]["batch_size"] * len(ctx)

    # Generate and save model
    checkpoint_prefix = os.path.join(checkpoint_dir, params_config["misc"]["checkpoint_prefix"])
    newest = None
    if os.path.exists(checkpoint_prefix + "-symbol.json"):
        for filename in os.listdir(checkpoint_dir):
            name, ext = os.path.splitext(filename)
            if ext != ".params":
                continue
            checkpoint_num = int(name.split("-")[1])
            if newest is None or newest < checkpoint_num:
                newest = checkpoint_num

        net = mx.gluon.nn.SymbolBlock.imports(symbol_file=checkpoint_prefix + "-symbol.json",
                                              input_names=["data0", "data1"],
                                              param_file=checkpoint_prefix + "-%04d.params" % newest,
                                              ctx=ctx)
        logger.info("Load model at Epoch[%d]", newest)
    else:
        net = MobileNetV2(config=params_config["mobilenet_v2"])
    net.hybridize()

    optimizer_cfg = params_config["train"]["optimizer"]
    lr_scheduler_cfg = params_config["train"]["lr_scheduler"]
    optimizer = optimizer_cfg["name"]
    optimizer_params = {
        "wd": optimizer_cfg["wd"],
        "learning_rate": optimizer_cfg["learning_rate"],
        "rescale_grad": 1.0 / multi_gpus_batch_size,
        "lr_scheduler": mx.lr_scheduler.FactorScheduler(step=lr_scheduler_cfg["step"],
                                                        factor=lr_scheduler_cfg["factor"],
                                                        stop_factor_lr=1e-6)
    }
    if optimizer == "sgd":
        optimizer_params.update({"momentum": optimizer_cfg["momentum"]})

    batch_end_callback = Speedometer(batch_size=multi_gpus_batch_size,
                                     frequent=params_config["train"]["log_frequent"],
                                     logger=logger)
    epoch_end_callback = CheckpointManager(path=checkpoint_dir,
                                           prefix=params_config["misc"]["checkpoint_prefix"],
                                           num_checkpoint=params_config["misc"]["num_checkpoint"],
                                           period=params_config["misc"]["checkpoint_period"],
                                           logger=logger)
    initializer = mx.init.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2)

    eval_metric = ["mse", AlongTrackError(name="ate"), CrossTrackError(name="cte"), Displacement(name="dpm")]
    loss = mx.gluon.loss.L2Loss()

    # set trainer
    trainer = Trainer(net=net,
                      train_dataset=train_data,
                      valid_dataset=valid_data,
                      batch_size=multi_gpus_batch_size,
                      shuffle=True,
                      ctx=ctx,
                      begin_epoch=params_config["train"]["begin_epoch"] if newest is None else newest + 1,
                      end_epoch=params_config["train"]["end_epoch"],
                      logger=logger)

    trainer.train(loss=loss,
                  eval_metric=eval_metric,
                  initializer=initializer,
                  epoch_end_callback=epoch_end_callback,
                  batch_end_callback=batch_end_callback,
                  optimizer=optimizer,
                  optimizer_params=optimizer_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="config/train.yaml",
                        help="Path of train config yaml file.")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0],
                        help="GPU to use, e.g. '0 1 2 3.'")
    parser.add_argument("--log", type=str, default="INFO",
                        help="Log level to console.")
    args = parser.parse_args()
    train(cfg_yaml=args.cfg,
          gpus=args.gpus,
          log_level=getattr(logging, args.log.upper(), logging.INFO))
