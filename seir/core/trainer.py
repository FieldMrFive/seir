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
    def __init__(self, net, train_dataset, batch_size, valid_dataset=None, shuffle=True,
                 ctx=None, begin_epoch=0, end_epoch=1000, logger=None):
        self._net = net
        self._ctx = ctx
        self._begin_epoch = begin_epoch
        self._end_epoch = end_epoch
        self._logger = logger

        self._train_data = None
        self._valid_data = None
        self.load_train_data(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)
        if valid_dataset is not None:
            self.load_valid_data(dataset=valid_dataset, batch_size=batch_size, shuffle=shuffle)

        if ctx is None:
            self._ctx = [mx.cpu()]
        if logger is None:
            self._logger = logging.getLogger()
            self._logger.setLevel(logging.INFO)

    def load_train_data(self, dataset, batch_size, shuffle=True):
        self._train_data = gluon.data.DataLoader(dataset=dataset,
                                                 batch_size=batch_size,
                                                 shuffle=shuffle,
                                                 last_batch='rollover')

    def load_valid_data(self, dataset, batch_size, shuffle=True):
        self._valid_data = gluon.data.DataLoader(dataset=dataset,
                                                 batch_size=batch_size,
                                                 shuffle=shuffle,
                                                 last_batch='rollover')

    def _train_batch(self, batch_data, batch_state, batch_label, loss, metric, trainer):
        data = gluon.utils.split_and_load(batch_data, self._ctx)
        label = gluon.utils.split_and_load(batch_label, self._ctx)
        state = gluon.utils.split_and_load(batch_state, self._ctx)
        with autograd.record():
            preds = [self._net(x1, x2) for x1, x2 in zip(data, state)]
            losses = [loss(y_hat, y) for y_hat, y in zip(preds, label)]

        for l in losses:
            l.backward()

        metric.update(labels=label, preds=preds)
        trainer.step(batch_data.shape[0])

    def _valid_once(self, batch_data, batch_state, batch_label, metric):
        data = gluon.utils.split_and_load(batch_data, self._ctx)
        label = gluon.utils.split_and_load(batch_label, self._ctx)
        state = gluon.utils.split_and_load(batch_state, self._ctx)

        preds = [self._net(x1, x2) for x1, x2 in zip(data, state)]
        mx.nd.waitall()
        metric.update(labels=label, preds=preds)

    def train(self, loss, eval_metric="acc", eval_frequence = 5, initializer=mx.init.Uniform(), epoch_end_callback=None,
              batch_end_callback=None, optimizer=None, optimizer_params=None, kvstore="local"):

        self._net.collect_params().initialize(initializer, self._ctx)

        trainer = gluon.Trainer(params=self._net.collect_params(),
                                optimizer=optimizer,
                                optimizer_params=optimizer_params,
                                kvstore=kvstore)

        if not isinstance(eval_metric, mx.metric.EvalMetric):
            eval_metric = mx.metric.create(eval_metric)

        for epoch in range(self._begin_epoch, self._end_epoch):
            tic = time.time()
            ts = time.time()
            for i, (data, state, label) in enumerate(self._train_data):
                # self._logger.info("load data cost %.4f", time.time() - ts)
                # ts = time.time()
                self._train_batch(data, state, label, loss, eval_metric, trainer)
                # self._logger.info("train batch cost %.4f", time.time() - ts)
                # ts = time.time()
                if batch_end_callback is not None:
                    batch_end_params = BatchEndParam(epoch=epoch, nbatch=i, eval_metric=eval_metric)
                    batch_end_callback(batch_end_params)
                # self._logger.info("batch callback cost %.4f", time.time() - ts)
                # ts = time.time()

            mx.nd.waitall()
            toc = time.time()
            self._logger.info("Epoch[%d] Time cost=%.3f", epoch, toc - tic)
            if epoch_end_callback is not None:
                epoch_end_callback(epoch, self._net)

            if self._valid_data is not None and epoch % eval_frequence == 0:
                self._logger.info("Validating........")
                for i, (data, state, label) in enumerate(self._train_data):
                    self._valid_once(data, state, label, eval_metric)
                name_value = eval_metric.get_name_value()
                msg = 'Epoch[%d] Validation ' + '\t%s=%f'*len(name_value)
                self._logger.info(msg, epoch, *sum(name_value, ()))
