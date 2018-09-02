import mxnet as mx

class Trainer(object):
    def __init__(self, net, ctx=None, begin_epoch=0, end_epoch=None):
        self.net = net
        self.ctx = ctx
        if ctx is None:
            self.ctx = [mx.cpu()]
        self.begin_epoch = begin_epoch
        self.end_epoch = end_epoch

    def train(self, dataset, loss, metrics, initializer, logger):
        self.net.collect_params().initialize(initializer, self.ctx)