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
    y_true_np = y_true.detach().numpy()
    y_pred_np = y_pred.detach().numpy()
    # NOTE: This will eliminate the multi-label nature of the data
    y_true_class = onehot_to_class(y_true_np)
    y_pred_class = onehot_to_class(y_pred_np)
    return metrics.balanced_accuracy_score(y_true_class, y_pred_class)


def average_precision(
    y_true: torch.Tensor, y_pred: torch.Tensor, average: str = "micro"
) -> float:
    """
        Computes the average precision score for the predictions
        Supports multi-label multi-class classification

        Parameters
        ----------
        y_true : torch.Tensor
            The true label tensor
        y_pred : torch.Tensor
            The predicted label tensor
        average : str, optional
            The type of averaging to be performed
            Look up scikit-learn multi-label classification for details
            Default value is micro

        Returns
        -------
        float
            The average precision score
    """
    y_true_np = y_true.detach().numpy()
    y_pred_np = y_pred.detach().numpy()
    return metrics.average_precision_score(y_true_np, y_pred_np, average=average)


def label_ranking(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
        Computes the ranking-based average precision for the predictions
        Supports multi-label multi-class classification

        Parameters
        ----------
        y_true : torch.Tensor
            The true label tensor
        y_pred : torch.Tensor
            The predicted label tensor

        Returns
        -------
        float
            The label ranking average precision score
    """
    y_true_np = y_true.detach().numpy()
    y_pred_np = y_pred.detach().numpy()
    return metrics.label_ranking_average_precision_score(y_true_np, y_pred_np)


def roc_auc_score(
    y_true: torch.Tensor, y_pred: torch.Tensor, average: str = "micro"
) -> float:
    """
        Computes the Area Under the Receiver Operating Characteristic Curve
        Supports multi-label multi-class classification

        Parameters
        ----------
        y_true : torch.Tensor
            The true label tensor
        y_pred : torch.Tensor
            The predicted label tensor
        average : str, optional
            The type of averaging to be performed
            Look up scikit-learn multi-label classification for details
            Default value is micro

        Returns
        -------
        float
            The label ranking average precision score
    """
    y_true_np = y_true.detach().numpy()
    y_pred_np = y_pred.detach().numpy()
    return metrics.roc_auc_score(y_true_np, y_pred_np, average=average)
