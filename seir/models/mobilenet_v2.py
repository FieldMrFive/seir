import mxnet.gluon.nn as nn

__all__ = ["MobileNetV2",
           "InvertedResidualBlock"]


def _add_conv(block, channels, kernel, stride=1, padding=0, groups=1, activate=True):
    block.add(nn.Conv2D(channels, kernel, stride, padding, groups=groups, use_bias=False))
    block.add(nn.BatchNorm())
    if activate:
        block.add(nn.Activation("relu"))


class MobileNetV2(nn.HybridBlock):
    def __init__(self, config, **kwargs):
        super(MobileNetV2, self).__init__(**kwargs)

        self.net_config = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]
        self.num_class = 5

        with self.name_scope():
            self.feature = nn.HybridSequential(prefix="feature_")
            with self.feature.name_scope():
                in_channels = 32
                _add_conv(self.feature, in_channels, 3, 2, 1)

                for seq in self.net_config:
                    self.feature.add(InvertedResidualBlock(in_channels=in_channels, out_channels=seq[1],
                                                           expansion=seq[0], stride=seq[3]))
                    for i in range(seq[2] - 1):
                        self.feature.add(InvertedResidualBlock(in_channels=seq[1], out_channels=seq[1],
                                                               expansion=seq[0], stride=1))
                    in_channels = seq[1]

                _add_conv(self.feature, 1280, 1)
                self.feature.add(nn.GlobalAvgPool2D())

            self.output = nn.HybridSequential(prefix="output_")
            with self.output.name_scope():
                self.output.add(nn.Conv2D(self.num_class, 1, use_bias=False, prefix="pred_"))
                self.output.add(nn.Flatten())

    def hybrid_forward(self, F, x):
        x = self.feature(x)
        x = self.output(x)
        return x


class InvertedResidualBlock(nn.HybridBlock):
    def __init__(self, in_channels, out_channels, expansion, stride, **kwargs):
        super(InvertedResidualBlock, self).__init__(**kwargs)
        self.short_cut = (stride == 1 and in_channels == out_channels)
        with self.name_scope():
            self.block = nn.HybridSequential()
            _add_conv(self.block, expansion * in_channels, 1)
            _add_conv(self.block, expansion * in_channels, 3, stride, padding=1, groups=expansion * in_channels)
            _add_conv(self.block, out_channels, 1, activate=False)

    def hybrid_forward(self, F, x):
        out = self.block(x)
        if self.short_cut:
            out = F.elemwise_add(out, x)
        return out
