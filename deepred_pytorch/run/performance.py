"""
    Module to evaluate model performance using various metrics
"""

import numpy as np
import sklearn.metrics as metrics
import torch


# Pytorch loss functions(?)
# True positives, False positives


def onehot_to_class(y: np.array) -> np.array:
    """
        Convert one-hot label encoding to class number enconding

        Parameters
        ----------
        y : np.array
            The label vector in one-hot enconding format

        Returns
        -------
        np.array
            The class number encoded label vector
            Class numbers start at 1
    """
    y_class = np.argmax(y, axis=1) + 1
    return y_class


def balanced_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
        Computes the accuracy score for the predictions

        Parameters
        ----------
        y_true : torch.Tensor
        y_pred : torch.Tensor

        Returns
        -------
        float
    """
    y_true_np = y_true.numpy()
    y_pred_np = y_pred.numpy()
    y_true_class = onehot_to_class(y_true_np)
    y_pred_class = onehot_to_class(y_pred_np)
    return metrics.balanced_accuracy_score(y_true_class, y_pred_class)
