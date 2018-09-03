import mxnet as mx
from collections import namedtuple
from mxnet import gluon
from mxnet import autograd
import time
import logging

BatchEndParam = namedtuple(
    'BatchEndParams', ['epoch', 'nbatch', 'eval_metric'])

__all__ = ["Trainer"]


class Trainer(object):
    def __init__(self, net, ctx=None, begin_epoch=0, end_epoch=1000, logger=None):
        self.net = net
        self.ctx = ctx
        self.begin_epoch = begin_epoch
        self.end_epoch = end_epoch
        self.logger = logger

        if ctx is None:
            self.ctx = [mx.cpu()]
        if logger is None:
            self.logger = logging.getLogger()
            self.logger.setLevel(logging.INFO)

    def train_batch(self, batch_data, batch_label, loss, metric, trainer):
        data = gluon.utils.split_and_load(batch_data, self.ctx)
        label = gluon.utils.split_and_load(batch_label, self.ctx)

        with autograd.record():
            preds = [self.net(X) for X in data]
            losses = [loss(Y_hat, Y) for Y_hat, Y in zip(preds, label)]

        for l in losses:
            l.backward()

        metric.update(preds, label)
        trainer.step(batch_data.shape[0])

    def train(self, dataset, batch_size, loss, eval_metric="acc", initializer=mx.init.Uniform(), shuffle=True,
              epoch_end_callback=None, batch_end_callback=None, optimizer=None, optimizer_params=None, kvstore="local"):

        self.net.collect_params().initialize(initializer, self.ctx)
        trainer = gluon.Trainer(params=self.net.collect_params(),
                                optimizer=optimizer,
                                optimizer_params=optimizer_params,
                                kvstore=kvstore)

        train_data = gluon.data.DataLoader(dataset=dataset,
                                           batch_size=batch_size,
                                           shuffle=shuffle)
        if not isinstance(eval_metric, mx.metric.EvalMetric):
            eval_metric = mx.metric.create(eval_metric)

        for epoch in range(self.begin_epoch, self.end_epoch):
            tic = time.time()
            for i, (data, label) in enumerate(train_data):
                self.train_batch(data, label, loss, eval_metric, trainer)

                if batch_end_callback is not None:
                    batch_end_params = BatchEndParam(epoch=epoch, nbatch=i, eval_metric=eval_metric)
                    batch_end_callback(batch_end_params)

            toc = time.time()
            self.logger.info("Epoch[%d] Time cost=%.3f", epoch, toc - tic)
            if epoch_end_callback is not None:
                epoch_end_callback(self.net)





