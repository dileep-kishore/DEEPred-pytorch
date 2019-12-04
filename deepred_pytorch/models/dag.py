"""
    Module that contains methods for creating and using the ModelDAG
"""

from collections import defaultdict
from itertools import product
import pathlib
from typing import Dict, Iterable, List, Set, Tuple

import networkx as nx
import numpy as np
import torch

from ..io import parse_go_dag
from ..io.data import parse_model_go_map

# ModelDAG.predict -> runs one forward pass through the DAG for one feature vector
# IF any model is None during predict then throw warning
# Have threshold as a param = 0.5 (any GO term >= this is added to the "stack")
# Return the set of GO terms


class ModelDAG:
    """
        The DAG that represents the complete DEEPred GO classifier

        Parameters
        ----------
        go_obo_file : pathlib.Path
            The file containing the GO term hierarchy
        model_go_map_dir : pathlib.Path
            Dir containing files where GO terms are associated with a model
            Each row of file must contain a GO term as the first entry

        Attributes
        ----------
        go_dag : nx.MultiDiGraph
            The DAG that represents the GO term hierarchy
        dag : nx.DiGraph
            The DAG that represents the complete DEEPred GO classifier
            Attributes
            ----------
            bipartite : int
                The bipartite label
            label_vector : List[str]
                The GO terms that form the label in order
            model : torch.nn.Module
                The `pytorch` NN classifier object
    """

    def __init__(
        self, go_obo_file: pathlib.Path, model_go_map_dir: pathlib.Path
    ) -> None:
        self.go_dag = parse_go_dag(go_obo_file)
        self.dag = self._build_dag(model_go_map_dir)
        self._remove_cycles()

    def _build_dag(self, model_go_map_dir: pathlib.Path) -> nx.DiGraph:
        """
            Build the bipartite DAG for the models and GO terms

            Parameters
            ----------
            model_go_map_dir : pathlib.Path
                Dir containing files where GO terms are associated with a model

            Returns
            -------
            nx.DiGraph
                The model-GO term bipartite DAG
        """
        dag = nx.DiGraph()
        model_go_dict: Dict[str, List[str]] = {}
        go_model_dict: Dict[str, Set[str]] = defaultdict(set)
        level_model_dict: Dict[int, Set[str]] = defaultdict(set)
        for model_go_map_file in model_go_map_dir.iterdir():
            go_terms = parse_model_go_map(model_go_map_file)
            model_name = model_go_map_file.stem
            if model_name in model_go_dict:
                raise ValueError("Duplicate models in model_go_map_dir")
            model_go_dict[model_name] = go_terms
            level = int(model_name.split("_")[1])
            level_model_dict[level].add(model_name)
            for go_term in go_terms:
                for go_parent in self.go_dag.predecessors(go_term):
                    go_model_dict[go_parent].add(model_name)
        # Add nodes to the dag
        dag.add_nodes_from(list(model_go_dict.keys()), bipartite=0, model=None)
        dag.add_nodes_from(self.go_dag.nodes, bipartite=1)
        # Add edges to the dag
        for level in range(1, len(level_model_dict) + 1):
            model_names = level_model_dict[level]
            go_terms_level: Set[str] = set()
            for model_name in model_names:
                go_terms = model_go_dict[model_name]
                go_terms_level.update(go_terms)
                dag.nodes[model_name]["label_vector"] = go_terms
                for go_term in go_terms:
                    dag.add_edge(model_name, go_term)
            next_level = level + 1
            if next_level in level_model_dict:
                for go_term in go_terms_level:
                    for model_name in go_model_dict[go_term]:
                        if model_name in level_model_dict[next_level]:
                            dag.add_edge(go_term, model_name)
        return dag

    def _remove_cycles(self) -> None:
        """ Remove cycles from the graph """
        cycles = nx.cycles.simple_cycles(self.dag)
        for path in cycles:
            # Remove self loops
            if len(path) == 2:
                u, v = path
                if self.dag.nodes[u]["bipartite"] == 0:
                    v, u = u, v
            # Cut bigger circuit
            else:
                for i, node in enumerate(path):
                    if self.dag.nodes[node]["bipartite"] == 1:
                        go_node_ind = i
                        break
                u, v = path[go_node_ind], path[go_node_ind + 1]
            u_node = self.dag.nodes[u]
            if u_node["bipartite"] == 1:
                self.dag.remove_edge(u, v)
        cycles = list(nx.cycles.simple_cycles(self.dag))
        assert len(cycles) == 0, "There are still cycles present"

    @property
    def model_nodes(self) -> Iterable[str]:
        """ Returns an iterable containing the model node names """
        for node, node_data in self.dag.nodes(data=True):
            if node_data["bipartite"] == 0:
                yield node

    @property
    def models_by_level(self) -> Iterable[Tuple[int, Set[str]]]:
        """ Returns an iterable containing the models at each level """
        level_model_dict: Dict[int, Set[str]] = defaultdict(set)
        for node, node_data in self.dag.nodes(data=True):
            if node_data["bipartite"] == 0:
                level = int(node.split("_")[1])
                level_model_dict[level].add(node)
        for level in range(1, len(level_model_dict) + 1):
            yield level, level_model_dict[level]

    def load_models(self, model_dir: pathlib.Path) -> None:
        """
            Loads the saved pytorch models onto the DAG

            Parameters
            ----------
            model_dir : pathlib.Path
                The directory containing the saved `pytorch` models
                The model names must match the model nodes in the DAG
        """
        for model_file in model_dir.iterdir():
            if model_file.suffix == ".pkl":
                model_name = model_file.stem
                model = torch.load(str(model_file))
                self.dag.nodes[model_name]["model"] = model

    def predict(self, feature_vector: torch.Tensor, threshold: float = 0.5) -> Set[str]:
        """
            Predict the GO terms associated with the feature vector

            Parameters
            ----------
            feature_vector : torch.Tensor
                A feature vector that represents one protein sequence
                Size: (D) where D - dimensions of the feature vector
            threshold : float, optional
                The threshold to be applied on the predicted probabilities

            Returns
            -------
            Set[str]
                The set of GO terms predicted
        """
        # Error handling
        for model_name in self.model_nodes:
            if self.dag.nodes[model_name]["model"] is None:
                raise ValueError("Please load all models before running predict")
        level_model_dict = dict(self.models_by_level)
        level = 1
        model_names = level_model_dict[level]
        all_go_terms: Set[str] = set()
        while len(model_names) > 0:
            selected_go_terms: Set[str] = set()
            for model_name in model_names:
                model = self.dag.nodes[model_name]["model"]
                model.eval()
                labels = self.dag.nodes[model_name]["label_vector"]
                prediction = model.predict(feature_vector).detach().numpy()
                prediction = prediction.reshape((prediction.shape[1]))
                prediction[prediction > threshold] = 1
                prediction[prediction < threshold] = 0
                try:
                    selected_inds = np.argwhere(prediction == 1)[0]
                except IndexError:
                    selected_inds = []
                selected_go_terms.update([labels[i] for i in selected_inds])
            next_models: Set[str] = set()
            for go_term in selected_go_terms:
                for model_name in nx.descendants(self.dag, go_term):
                    if model_name in level_model_dict[level + 1]:
                        next_models.add(model_name)
            all_go_terms.update(selected_go_terms)
            level = level + 1
            model_names = next_models
        return all_go_terms

    def predict_v2(
        self, feature_vector: torch.Tensor, threshold: float = 0.5
    ) -> Set[str]:
        """
            Predict the GO terms associated with the feature vector

            feature_vector : torch.Tensor
                A feature vector that represents one protein sequence
                Size: (D) where D - dimensions of the feature vector
            threshold : float, optional
                The threshold to be applied on the predicted probabilities

            Returns
            -------
            Set[str]
                The set of GO terms predicted
        """
        full_go_dag = self.go_dag
        # Error handling
        for model_name in self.model_nodes:
            if self.dag.nodes[model_name]["model"] is None:
                raise ValueError("Please load all models before running predict")
        level_model_dict = dict(self.models_by_level)
        go_prob_dict_list: List[Dict[str, float]] = []
        for level, model_names in level_model_dict.items():
            go_prob_dict_list.append(defaultdict(float))
            for model_name in model_names:
                model = self.dag.nodes[model_name]["model"]
                model.eval()
                labels = self.dag.nodes[model_name]["label_vector"]
                prediction = model.predict(feature_vector).detach().numpy()
                prediction = prediction.reshape((prediction.shape[1]))
                try:
                    selected_inds = np.argwhere(prediction > threshold)[0]
                    selected_probs = prediction[selected_inds]
                except IndexError:
                    selected_inds = []
                    selected_probs = []
                predicted_go_terms = [labels[i] for i in selected_inds]
                for go_term, prob in zip(predicted_go_terms, selected_probs):
                    go_prob_dict_list[-1][go_term] += prob
        go_dag = nx.DiGraph()
        for ind, go_prob_dict in enumerate(go_prob_dict_list):
            connect_flag = False
            level = ind + 1
            if level > 1:
                connect_flag = True
            for go_term, prob in go_prob_dict.items():
                assert go_term in full_go_dag.nodes
                go_dag.add_node(go_term, prob=prob)
            if connect_flag:
                go_prob_dict_parent = go_prob_dict_list[ind - 1]
                go_terms_parent = list(go_prob_dict_parent.keys())
                go_terms_current = list(go_prob_dict.keys())
                for u, v in product(go_terms_parent, go_terms_current):
                    if (u, v) in full_go_dag.edges:
                        go_dag.add_edge(u, v)
        source_go_terms = list(go_prob_dict_list[0].keys())
        target_go_terms = list(go_prob_dict_list[-1].keys())
        path_scores: List[Tuple[list, float]] = []
        for source, target in product(source_go_terms, target_go_terms):
            try:
                path = nx.shortest_path(go_dag, source=source, target=target)
                score = sum([go_dag.nodes[n]["prob"] for n in path])
            except:
                path = []
                score = 0
            path_scores.append((path, score))
        sorted_path_scores = sorted(path_scores, key=lambda x: x[1], reverse=True)
        return set(sorted_path_scores[0][0])
