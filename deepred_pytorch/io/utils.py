"""
    Module contains utility functions for processing data
"""

from typing import Any, Iterable, Tuple
from warnings import warn

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch


Batch = Tuple[int, torch.Tensor, torch.Tensor]


def normalize_data(data: torch.Tensor, scaler: Any = None) -> Tuple[torch.Tensor, Any]:
    """
        Normalize data using the provided scaler object
        Uses `StandardScaler` from scikit-learn if scaler not provided

        Parameters
        ----------
        data : torch.Tensor
            The data to be normalized
            Each sample must be in one row
            Features must be along the columns
        scaler : object, optional
            Object used to normalize the data
            Must have the `.transform` method

        Returns
        -------
        data_normalized
            The normalized data
        scaler
            The scaler object used to normalize the data
    """
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(data)
    data_normalized = torch.Tensor(scaler.transform(data))
    return data_normalized, scaler


def split_batch(
    x: torch.Tensor, y: torch.Tensor, minibatch_size: int
) -> Iterable[Batch]:
    """
        Split batch into mini-batches
        x and y must have the same number of samples

        Parameters
        ----------
        x : torch.Tensor
            Feature tensor for the entire batch
        y : torch.Tensor
            Label tensor for the entire batch
        minibatch_size : int
            The size of the minibatch

        Returns
        -------
        Iterable[Batch]
    """
    assert x.shape[0] == y.shape[0], "x and y shapes do not match"
    n_samples = x.shape[0]
    if n_samples % minibatch_size != 0:
        warn("Final batch will have fewer samples")
    for batch_ind, ind in enumerate(range(0, n_samples, minibatch_size)):
        x_split = x[ind : ind + minibatch_size, :]
        y_split = y[ind : ind + minibatch_size, :]
        yield batch_ind, x_split, y_split


def shuffle_data(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
        Shuffle the order of the samples in the dataset

        Parameters
        ----------
        x : torch.Tensor
            Feature tensor for the entire batch
        y : torch.Tensor
            Label tensor for the entire batch

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The shuffled dataset
    """
    assert x.shape[0] == y.shape[0], "x and y shapes do not match"
    inds = list(range(x.shape[0]))
    np.random.shuffle(inds)
    x_shuffled = x[inds, :]
    y_shuffled = y[inds, :]
    return x_shuffled, y_shuffled
