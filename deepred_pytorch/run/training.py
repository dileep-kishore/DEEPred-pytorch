"""
    Module to train the DEEPred NN classifier
"""

import torch
from torch.utils.data import RandomSampler
import torch.nn as nn
import torch.optim as optim

from ..models import Model
from ..io.utils import split_batch, shuffle_data


def train(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int,
    parameters: dict,
    minibatch_size: int = 32,
    batchnorm: bool = True,
    p_dropout: float = 0.5,
    learning_rate: float = 0.005,
) -> nn.Module:
    """
        Train the DEEPred pytroch classifier

        Parameters
        ----------
        x_train : torch.Tensor
            The tensor containing the feature vectors for training
        y_train : torch.Tensor
            The tensor containing the label vectors for training
        epochs : int
            The number of epochs for which to train the model
        parameters : dict
            The parameters that define the nodes of the neural network
        minibatch_size : int, optional
            The size of the minibatches created during training
            Default value is 32
        batchnorm : bool, optional
            Flag that turns on batch normalization of the hidden layers
            Default value is true
        p_dropout : float, optional
            The probability of the dropout
            Default value is 0.5
        learning_rate : float, optional
            The learning rate for the neural network
            Default value is 0.005

        Returns
        -------
        List[str]
            The list of protein ids associated with the GO term
    """
    # NOTE: Normalize data before passing it into this function
    model = Model(parameters, batchnorm=batchnorm, p_dropout=p_dropout)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(epochs):
        loss_per_epoch = 0.0
        x_shuff, y_shuff = shuffle_data(x_train, y_train)
        for batch_ind, x_mb, y_mb in split_batch(x_shuff, y_shuff, minibatch_size):
            if x_mb.shape[0] < 5:
                continue
            optimizer.zero_grad()
            y_pred = model(x_mb)
            loss = criterion(y_pred, y_mb)
            loss.backward()
            loss_per_epoch += loss.item()
            optimizer.step()
        print(f"Epoch {epoch}: train loss: {loss_per_epoch}")
    return model
