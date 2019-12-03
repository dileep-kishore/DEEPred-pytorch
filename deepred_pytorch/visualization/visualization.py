"""
    Module that contains functions for visualizing model performance
"""

import re
import pathlib
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..io.data import parse_model_output

sns.set(style="ticks")


def create_perf_dataframe(model_output_dir: pathlib.Path) -> pd.DataFrame:
    """
        Create a pandas dataframe that tabulates model performance and statistics

        Parameters
        ----------
        model_output_dir : pathlib.Path
            The directory containing the model run output

        Returns
        -------
        pd.DataFrame
    """
    pattern = re.compile(r"MFGOTerms30_(.)_(.*)_(.*)_(.*)_output")
    data_list: List[dict] = []
    for model_output_file in model_output_dir.glob("*_output.txt"):
        model_name = model_output_file.stem
        match = re.match(pattern, model_name)
        if match:
            level, size_lb, size_ub, model_num = match.groups()
        else:
            raise ValueError("Model format is incorrect")
        accuracy_dict = parse_model_output(model_output_file)
        data_list.append(
            {
                **accuracy_dict,
                "model_size": f"{size_lb}-{size_ub}",
                "level": int(level),
                "model_num": int(model_num),
            }
        )
    data_df = pd.DataFrame(data_list)
    return data_df


def perf_vs_modelsize(
    model_output_dir: pathlib.Path, fig_path: pathlib.Path
) -> pd.DataFrame:
    """
        Visualization model performance vs. model size

        Parameters
        ----------
        model_output_dir : pathlib.Path
            The directory containing the model run output
        fig_path : pathlib.Path
            The path to save the figure

        Returns
        -------
        pd.DataFrame
            DataFrame containing model performance
    """
    plt.figure()
    data_df = create_perf_dataframe(model_output_dir)
    x_order = sorted(set(data_df["model_size"]), key=lambda x: int(x.split("-")[0]))
    sns.boxplot(
        x="model_size", y="roc_auc", data=data_df, order=x_order, palette="Set2",
    )
    sns.swarmplot(x="model_size", y="roc_auc", data=data_df, color=".25", order=x_order)
    plt.xlabel("Model size")
    plt.ylabel("ROC AUC")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(fig_path)


def perf_vs_level(
    model_output_dir: pathlib.Path, fig_path: pathlib.Path
) -> pd.DataFrame:
    """
        Visualization of model performance vs. GO level

        Parameters
        ----------
        model_output_dir : pathlib.Path
            The directory containing the model run output
        fig_path : pathlib.Path
            The path to save the figure

        Returns
        -------
        pd.DataFrame
            DataFrame containing model performance
    """
    plt.figure()
    data_df = create_perf_dataframe(model_output_dir)
    sns.boxplot(x="level", y="roc_auc", data=data_df, palette="Set1")
    sns.swarmplot(x="level", y="roc_auc", data=data_df, color=".25")
    plt.xlabel("GO level")
    plt.ylabel("ROC AUC")
    plt.tight_layout()
    plt.savefig(fig_path)
