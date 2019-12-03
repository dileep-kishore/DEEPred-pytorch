"""
    Module that contains functions for visualizing model performance
"""

import re
import pathlib
from typing import List

import pandas as pd
import seaborn as sns

from ..io.data import parse_model_output


def perf_vs_modelsize(
    model_output_dir: pathlib.Path, fig_path: pathlib.Path
) -> pd.DataFrame:
    """
        Visualization model performance vs. model size

        Parameters
        ----------
        model_output_dir : pathlib.Path
        fig_path : pathlib.Path

        Returns
        -------
        pd.DataFrame
            DataFrame containing model performance
    """
    pattern = re.compile(r"MFGOTerms30_(.)_(.*)_(.*)_(.)_output.txt")
    data_list: List[dict] = []
    for model_output_file in model_output_dir.glob("*_output.txt"):
        match = re.match(pattern, str(model_output_file))
        if match:
            level, size_lb, size_ub, model_num = match.groups()
        else:
            raise ValueError("Model format is incorrect")
        accuracy_dict = parse_model_output(model_output_file)
        data_list.append({**accuracy_dict, "model_size": f"{size_lb}-{size_ub}"})
    df = pd.DataFrame(data_list)
    print(df)
