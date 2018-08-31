#!/usr/bin/env python

import argparse
import logging
import os
import yaml
import mxnet as mx
from datetime import datetime
from pprint import pformat


def training(cfg_yaml, gpus, log_level):
    params_config = load_yaml(cfg_yaml)

    # set up logs and checkpoint dir
    checkpoint_dir = os.path.join(
        params_config["misc"]["checkpint_dir"], params_config["training"]["prefix"],
        datetime.now().strftime("%Y_%m_%d_%H_%M"))
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # set up logger
    log_path = os.path.join(checkpoint_dir, "log")
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
    train_iter = TrajIter(config=params_config, num_gpus=len(gpus))
    multi_gpus_batch_size = params_config["training"]["batch_size"] * len(ctx)

    # load symbol
    symbol = get_mobilenetv2_symbol(num_classes=10)

    optimizer_cfg = params_config["training"]["optimizer"]
    lr_scheduler_cfg = params_config["training"]["lr_scheduler"]
    optimizer = optimizer_cfg["name"]
    optimizer_params = {
        'wd': optimizer_cfg['wd'],
        'learning_rate': optimizer_cfg['learning_rate'],
        'rescale_grad': 1.0 / multi_gpus_batch_size,
        'lr_scheduler': mx.lr_scheduler.FactorScheduler(step=lr_scheduler_cfg['step'],
                                                        factor=lr_scheduler_cfg['factor'],
                                                        stop_factor_lr=1e-6)
    }
    if optimizer == 'sgd':
        optimizer_params.update({'momentum': optimizer_cfg['momentum']})

    # set solver
    arg_params, aux_params = None, None
    model = Solver(symbol, ctx=ctx,
                   begin_epoch=params_config['training']['begin_epoch'],
                   end_epoch=params_config['training']['end_epoch'],
                   arg_params=arg_params,
                   aux_params=aux_params)
    batch_end_callback = mx.callback.Speedometer(batch_size=params_config['training']['batch_size'],
                                                 frequent=params_config['training']['frequent'])
    epoch_end_callback = mx.callback.do_checkpoint(prefix=os.path.join(checkpoint_dir,
                                                                       params_config['training']['prefix']))
    initializer = mx.init.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2)

    eval_metric = ['acc', 'mse']

    model.fit(prefix=params_config['training']['prefix'],
              train_data=train_iter,
              eval_metric=eval_metric,
              batch_end_callback=batch_end_callback,
              epoch_end_callback=epoch_end_callback,
              initializer=initializer,
              optimizer=optimizer,
              optimizer_params=optimizer_params,
              )


def load_yaml(path):
    with open(path) as fin:
        config = yaml.load(fin)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="config/train.yaml",
                        help="Path of train config yaml file.")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0],
                        help="GPU to use, e.g. '0 1 2 3.'")
    parser.add_argument("--log", type=str, default="INFO",
                        help="Log level to console.")
    args = parser.parse_args()
    training(cfg_yaml=args.cfg,
             gpus=args.gpus,
             log_level=getattr(logging, args.log.upper(), logging.INFO))
