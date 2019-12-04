#!/usr/bin/env python3

import pathlib

from deepred_pytorch.visualization import perf_vs_modelsize, perf_vs_level


def main(model_output_dir, fig_path):
    perf_vs_modelsize(model_output_dir, fig_path / "perf_vs_modelsize.png")
    perf_vs_level(model_output_dir, fig_path / "perf_vs_level.png")


if __name__ == "__main__":
    MODEL_OUTPUT_DIR = pathlib.Path("outputs/MF_CTriad")
    FIG_PATH = pathlib.Path(".")
    main(MODEL_OUTPUT_DIR, FIG_PATH)
