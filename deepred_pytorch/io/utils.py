"""
    Module contains utility functions for processing data
"""

from typing import Any, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler


def normalize_data(data: np.array, scaler: Any = None) -> Tuple[np.array, Any]:
    """
        Normalize data using the provided scaler object
        Uses `StandardScaler` from scikit-learn if scaler not provided

        Parameters
        ----------
        data : np.array
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
    data_normalized = scaler.tranform(data)
    return data_normalized, scaler
