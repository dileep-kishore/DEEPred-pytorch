"""
    Script to train 1 model
    Each model handles 3-5 GO terms from one level of the GO DAG
"""

import pathlib

import torch

from deepred_pytorch.io import parse_data
from deepred_pytorch.io.utils import normalize_data
from deepred_pytorch.run import train
from deepred_pytorch.run.performance import balanced_accuracy


def main(
    feature_vector_file, model_go_map_file, go_prot_map_train_dir, go_prot_map_test_dir
):
    x_train, y_train = parse_data(
        feature_vector_file, model_go_map_file, go_prot_map_train_dir
    )
    parameters = {
        "n_x": x_train.shape[1],
        "n_h1": 1400,
        "n_h2": 100,
        "n_y": y_train.shape[1],
    }
    x_train_tensor = torch.FloatTensor(x_train.values)
    y_train_tensor = torch.FloatTensor(y_train.values)
    x_train_tensor, scaler = normalize_data(x_train_tensor)
    model = train(
        x_train_tensor,
        y_train_tensor,
        epochs=1000,
        parameters=parameters,
        minibatch_size=32,
        batchnorm=True,
        p_dropout=0.5,
    )
    x_test, y_test = parse_data(
        feature_vector_file, model_go_map_file, go_prot_map_test_dir
    )
    x_test_tensor = torch.FloatTensor(x_test.values)
    y_test_tensor = torch.FloatTensor(y_test.values)
    x_test_tensor, _ = normalize_data(x_test_tensor, scaler=scaler)
    y_test_pred_tensor = model(x_test_tensor)
    acc = balanced_accuracy(y_test_tensor, y_test_pred_tensor)
    print(f"Accuracy = {acc}")


if __name__ == "__main__":
    FEATURE_VECTOR_FILE = pathlib.Path(
        "data/feature_vectors/CTriad_protr_training_MF.csv"
    )
    MODEL_GO_MAP_FILE = pathlib.Path("data/model_go_map/MFGOTerms30_1_1001_2000_1.txt")
    GO_PROT_MAP_TRAIN_DIR = pathlib.Path("data/go_prot_map/training")
    GO_PROT_MAP_TEST_DIR = pathlib.Path("data/go_prot_map/testing")
    main(
        FEATURE_VECTOR_FILE,
        MODEL_GO_MAP_FILE,
        GO_PROT_MAP_TRAIN_DIR,
        GO_PROT_MAP_TEST_DIR,
    )