"""
    Module that contains methods for creating and using the ModelDAG
"""

from collections import defaultdict
import pathlib
from typing import Dict, List, Set

import networkx as nx
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
