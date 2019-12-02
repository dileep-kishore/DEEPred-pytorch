"""
    The DEEPred neural network classifer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
        The neural network model to classify protein sequences
        into relevant GO term annotations
        This model closely resembles the DEEPred architecture

        Parameters
        ----------
        parameters : dict
            The parameters that define the nodes of the neural network
        batchnorm : bool, optional
            Flag that turns on batch normalization of the hidden layers
            Default value is true
        p_dropout : float, optional
            The probability of the dropout
            Default value is 0.5
    """

    def __init__(
        self, parameters: dict, batchnorm: bool = True, p_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        n_x = parameters["n_x"]
        n_h1 = parameters["n_h1"]
        n_h2 = parameters["n_h2"]
        n_y = parameters["n_y"]
        self._l1 = nn.Linear(n_x, n_h1)
        self._l2 = nn.Linear(n_h1, n_h2)
        self._l3 = nn.Linear(n_h2, n_y)
        self._dropout = nn.Dropout(p=p_dropout)
        self._batchnorm = batchnorm
        if self._batchnorm:
            self._batchnorm1 = nn.BatchNorm1d(n_h1)
            self._batchnorm2 = nn.BatchNorm1d(n_h2)

    def forward(self, x):
        """ Forward propagation of the model """
        z_1 = self._l1(x)
        if self._batchnorm:
            bn_1 = self._batchnorm1(z_1)
            a_1 = F.relu(bn_1)
        else:
            a_1 = F.relu(z_1)
        a_1 = self._dropout(a_1)
        z_2 = self._l2(a_1)
        if self._batchnorm:
            bn_2 = self._batchnorm2(z_2)
            a_2 = F.relu(bn_2)
        else:
            a_2 = F.relu(z_2)
        a_2 = self._dropout(a_2)
        z_3 = self._l3(a_2)
        a_3 = F.sigmoid(z_3)
        return a_3
