"""
    Module that contains functions for loading and parsing data
"""

from collections import defaultdict
import pathlib
from typing import Dict, List, Set, Tuple
from warnings import warn

import obonet
import networkx as nx
import numpy as np
import pandas as pd


def parse_model_go_map(model_go_map_file: pathlib.Path) -> List[str]:
    """
        Parses the model_go_map file
        Returns the GO terms associated with the model

        Parameters
        ----------
        model_go_map_file : pathlib.Path
            File containing the GO terms associated with the model to be trained
            Each row must contain a GO term as the first entry

        Returns
        -------
        List[str]
            The list of GO terms associated with the model
    """
    model_go_map = pd.read_table(model_go_map_file, header=None, index_col=0)
    return list(model_go_map.index)


def parse_go_prot_map(go_prot_map_file: pathlib.Path) -> List[str]:
    """
        Parses the go_prot_map file
        Returns the list of uniprot protein ids associated with the GO term

        Parameters
        ----------
        go_prot_map_file : pathlib.Path
            File containing uniprot protein ids for the GO term
            Each file name must contain the GO term and each line must be a prot id

        Returns
        -------
        List[str]
            The list of protein ids associated with the GO term
    """
    with open(go_prot_map_file) as fid:
        prot_list = fid.read().split("\n")
    return prot_list


def parse_data(
    feature_vector_file: pathlib.Path,
    model_go_map_file: pathlib.Path,
    go_prot_map_dir: pathlib.Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
        Parse training data for DEEPred

        Parameters
        ----------
        feature_vector_file : pathlib.Path
            File containing the feature vectors
            Row indices must be uniprot protein ids
        model_go_map_file : pathlib.Path
            File containing the GO terms associated with the model to be trained
            Each row must contain a GO term as the first entry
        go_prot_map_dir : pathlib.Path
            Directory containing the files that map GO terms to uniprot protein ids
            Each file name must contain the GO term and each line must be a prot id

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            feature_vectors dataframe with each row containing a feature vector for a prot
            prot_labels dataframe with each row containg labels for a prot
    """
    feature_vectors = pd.read_csv(feature_vector_file, index_col=0)
    go_terms = parse_model_go_map(model_go_map_file)
    prot_go_map: Dict[str, Set[str]] = defaultdict(set)
    n_labels = len(go_terms)
    label_shape = (feature_vectors.shape[0], n_labels)
    prot_labels = pd.DataFrame(
        np.zeros(label_shape), index=feature_vectors.index, columns=go_terms, dtype=int
    )
    for go_term in go_terms:
        for go_prot_map_file in go_prot_map_dir.iterdir():
            if go_term in str(go_prot_map_file):
                prot_ids = parse_go_prot_map(go_prot_map_file)
                for prot_id in prot_ids:
                    if prot_id in feature_vectors.index:
                        prot_labels.loc[prot_id, go_term] = 1
                        prot_go_map[prot_id].add(go_term)
                        if len(prot_go_map[prot_id]) > 1:
                            warn(
                                "There is more than one GO term assocated with a protein id"
                            )
                break
    prots_to_drop = prot_labels.index[prot_labels.sum(axis=1) == 0]
    feature_vectors.drop(prots_to_drop, inplace=True)
    prot_labels.drop(prots_to_drop, inplace=True)
    print(f"Number of samples in dataset: {feature_vectors.shape[0]}")
    return feature_vectors, prot_labels


def parse_go_dag(go_obo_file: pathlib.Path) -> nx.MultiDiGraph:
    """
        Parse the GO obo file into a DAG

        Parameters
        ----------
        go_obo_file : pathlib.Path
            The file containing the GO term hierarchy

        Returns
        -------
        nx.MultiGraph
            The GO DAG as a networkx `MultiDiGraph`
    """
    go_dag = obonet.read_obo(str(go_obo_file))
    assert nx.is_directed_acyclic_graph(go_dag), "The graph must be a DAG"
    return go_dag


def parse_prot_go_map(go_prot_map_dir: pathlib.Path) -> Dict[str, Set[str]]:
    """
        Parse all go_prot_map files
        Returns dictionary mapping proteins to set of GO terms

        Parameters
        ----------
        go_prot_map_dir : pathlib.Path
            Dir containing files with uniprot protein ids for every GO term
            Each file name must contain the GO term and each line must be a prot id

        Returns
        -------
        Dict[str, Set[str]]
            Mapping from proteins to its associated set of GO terms
    """
    prot_go_map: Dict[str, Set[str]] = defaultdict(set)
    for go_prot_map_file in go_prot_map_dir.iterdir():
        go_term = go_prot_map_file.stem
        prot_list = parse_go_prot_map(go_prot_map_file)
        for protein in prot_list:
            prot_go_map[protein].add(go_term)
    return prot_go_map


def parse_model_output(model_output_file: pathlib.Path) -> Dict[str, float]:
    """
        Parse the model output file and return a dictionary of accuracies

        Parameters
        ----------
        model_output_file : pathlib.Path
            The model output file
            The last line contains the accuracy information

        Returns
        -------
        Dict[str, float]
            Dictionary of accurracy and corresponding value
            keys in avg_precision, lrap, roc_auc
    """
    accuracy_dict: Dict[str, float] = {
        "avg_precision": 0,
        "lrap": 0,
        "roc_auc": 0,
    }
    with open(model_output_file) as fid:
        data = fid.readlines()[-1]
    accuracies = [float(d.split(" = ")[-1]) for d in data.split(",")]
    accuracy_dict["avg_precision"] = accuracies[0]
    accuracy_dict["lrap"] = accuracies[1]
    accuracy_dict["roc_auc"] = accuracies[2]
    return accuracy_dict
