#!/usr/bin/env python3

import pathlib

import matplotlib.pyplot as plt
import networkx as nx
import torch

from deepred_pytorch.models import ModelDAG
from deepred_pytorch.io import parse_data
from deepred_pytorch.io.data import parse_prot_go_map
from deepred_pytorch.io.utils import normalize_data


def main(
    go_obo_file,
    model_go_map_dir,
    model_dir,
    feature_vector_file,
    model_go_map_file,
    go_prot_map_train_dir,
    go_prot_map_test_dir,
):
    x_train, y_train = parse_data(
        feature_vector_file, model_go_map_file, go_prot_map_train_dir
    )
    x_train_tensor = torch.FloatTensor(x_train.values)
    # y_train_tensor = torch.FloatTensor(y_train.values)
    y_true_labels_train = parse_prot_go_map(go_prot_map_train_dir)
    x_train_tensor, scaler = normalize_data(x_train_tensor)
    x_test, y_test = parse_data(
        feature_vector_file, model_go_map_file, go_prot_map_test_dir
    )
    assert (y_train.columns == y_test.columns).all()
    x_test_tensor = torch.FloatTensor(x_test.values)
    # y_test_tensor = torch.FloatTensor(y_test.values)
    y_true_labels_test = parse_prot_go_map(go_prot_map_test_dir)
    y_true_labels = {**y_true_labels_train, **y_true_labels_test}
    x_test_tensor, _ = normalize_data(x_test_tensor, scaler=scaler)

    md = ModelDAG(go_obo_file, model_go_map_dir)
    md.load_models(model_dir)
    for ind in range(x_test_tensor.shape[0]):
        prot_name = x_test.index[ind]
        sample = x_test_tensor[ind, :]
        predicted_go_terms = md.predict(sample.view(1, sample.shape[0]), threshold=0.75)
        true_go_terms = y_true_labels[prot_name]
        print(f"True: {true_go_terms}, Predicted: {predicted_go_terms}")
        jacc_score = len(predicted_go_terms & true_go_terms) / len(
            predicted_go_terms | true_go_terms
        )
        print(f"Jaccard Score: {jacc_score}")


if __name__ == "__main__":
    GO_OBO_FILE = pathlib.Path("data/GO_db/go-basic.obo")
    MODEL_GO_MAP_DIR = pathlib.Path("data/model_go_map")
    MODEL_DIR = pathlib.Path("outputs/MF_CTriad")

    FEATURE_VECTOR_FILE = pathlib.Path("data/feature_vectors/CTriad_training_MF.csv")
    # all this does is limit the number of proteins in the train or test set
    # MODEL_GO_MAP_FILE = pathlib.Path("data/model_go_map_all.txt")
    MODEL_GO_MAP_FILE = pathlib.Path("data/model_go_map/MFGOTerms30_1_1001_2000_1.txt")
    GO_PROT_MAP_TRAIN_DIR = pathlib.Path("data/go_prot_map/training")
    GO_PROT_MAP_TEST_DIR = pathlib.Path("data/go_prot_map/testing")

    main(
        GO_OBO_FILE,
        MODEL_GO_MAP_DIR,
        MODEL_DIR,
        FEATURE_VECTOR_FILE,
        MODEL_GO_MAP_FILE,
        GO_PROT_MAP_TRAIN_DIR,
        GO_PROT_MAP_TEST_DIR,
    )
