"""
    Module that contains functions for loading and parsing data
"""

from collections import defaultdict
import pathlib
from typing import Dict, List, Set, Tuple
from warnings import warn

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
    model_go_map = pd.read_table(model_go_map_file, header=False, index_col=0, sep=",")
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
        go_prot_map = fid.readlines()
    return go_prot_map


def create_training_data(
    feature_vector_file: pathlib.Path,
    model_go_map_file: pathlib.Path,
    go_prot_map_dir: pathlib.Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
        Create training data for DEEPred

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
    feature_vectors = pd.read_csv(feature_vector_file, header=True, index_col=0)
    go_terms = parse_model_go_map(model_go_map_file)
    # NOTE: We are assuming there is only one GO term associated with a prot id
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
    warn(f"Dropping proteins {prots_to_drop}")
    return feature_vectors, prot_labels
