import numpy as np
import mxnet as mx
from mxnet.metric import EvalMetric, check_label_shapes

__all__ = ["AlongTrackError",
           "CrossTrackError",
           "Displacement"]


def _get_track_heading(predict):
    direct = np.zeros(predict.shape)
    direct[:, 0] = predict[:, 1] - predict[:, 0]
    for i in range(1, predict.shape[1]):
        direct[:, i] = predict[:, i] - predict[:, i - 1]
    return np.arctan2(direct[:, :, 1], direct[:, :, 0])


def _reshape_data(data_lst):
    np_lst = [data.asnumpy() for data in data_lst]
    for data in np_lst:
        assert len(data.shape) == 2 and data.shape[1] % 2 == 0, "Shape of label and predict should be (n, 2*m)"
    return [data.reshape(data.shape[0], -1, 2) for data in np_lst]


class AlongTrackError(EvalMetric):
    def __init__(self, name='along_track_error',
                 output_names=None, label_names=None):
        super(AlongTrackError, self).__init__(
            name, output_names=output_names, label_names=label_names)

    def update(self, labels, preds):
        labels, preds = check_label_shapes(labels, preds, True)

        for label, pred in zip(labels, preds):
            label, pred = _reshape_data([label, pred])

            angle = _get_track_heading(pred)
            error = label - pred
            error[:, :, 0] *= np.cos(angle)
            error[:, :, 1] *= np.sin(angle)

            self.sum_metric += np.abs(np.sum(error, axis=2)).mean()
            self.num_inst += 1


class CrossTrackError(EvalMetric):
    def __init__(self, name='cross_track_error',
                 output_names=None, label_names=None):
        super(CrossTrackError, self).__init__(
            name, output_names=output_names, label_names=label_names)

    def update(self, labels, preds):
        labels, preds = check_label_shapes(labels, preds, True)

        for label, pred in zip(labels, preds):
            label, pred = _reshape_data([label, pred])

            angle = _get_track_heading(pred)
            error = label - pred
            error[:, :, 0] *= np.sin(angle)
            error[:, :, 1] *= -np.cos(angle)

            self.sum_metric += np.abs(np.sum(error, axis=2)).mean()
            self.num_inst += 1


class Displacement(EvalMetric):
    def __init__(self, name='displacement',
                 output_names=None, label_names=None):
        super(Displacement, self).__init__(
            name, output_names=output_names, label_names=label_names)

    def update(self, labels, preds):
        labels, preds = check_label_shapes(labels, preds, True)

        for label, pred in zip(labels, preds):
            label, pred = _reshape_data([label, pred])
            error = label - pred

            self.sum_metric += np.abs(np.linalg.norm(error, axis=2)).mean()
            self.num_inst += 1


if __name__ == "__main__":
    target = mx.nd.array([[1, 3, 2, 4, 3, 5],
                          [-1, 3, -2, 4, -3, 5]])
    predict = mx.nd.array([[1, 1, 2, 2, 3, 3],
                         [-1, 1, -2, 2, -3, 3]])

    along_track_error = AlongTrackError()
    along_track_error.update(labels=target, preds=predict)
    cross_track_error = CrossTrackError()
    cross_track_error.update(labels=target, preds=predict)
    displacement = Displacement()
    displacement.update(labels=target, preds=predict)
    print(along_track_error.get())
    print(cross_track_error.get())
    print(displacement.get())
