import numpy as np

from nn import Module
import functional as F


class CrossEntropyLoss(Module):

    def __init__(self, model):
        self._model = model

    def forward(self, y_pred, labels):
        predictions = F.softmax(y_pred)
        self._predictions = predictions
        self._labels = labels
        loss = np.dot(labels, predictions)
        return loss

    def backward(self):
        out_grad = self._labels - self._predictions
        self._model.backward(out_grad)
