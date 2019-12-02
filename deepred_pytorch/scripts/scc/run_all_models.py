#!/usr/bin/env python3

import pathlib
import subprocess


def main(model_data, model_output_dir):
    for model_file in model_data:
        mname = model_file.stem
        output_file = str(model_output_dir / (mname + "_output.txt"))
        error_file = str(model_output_dir / (mname + "_error.txt"))
        args = ["qsub", "-o", output_file, "-e", error_file, "run_models.sh", str(model_file)]
        print(" ".join(args))
        # subprocess.Popen(args)


if __name__ == "__main__":
    MODEL_DATA = pathlib.Path("data/model_go_map").glob("*.txt")
    MODEL_OUTPUT_DIR = pathlib.Path("data/model_outputs")
    main(MODEL_DATA, MODEL_OUTPUT_DIR)
