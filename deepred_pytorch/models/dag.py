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
        dag : nx.MultiDiGraph
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

    def _build_dag(self, model_go_map_dir: pathlib.Path) -> nx.MultiDiGraph:
        """
            Build the bipartite DAG for the models and GO terms

            Parameters
            ----------
            model_go_map_dir : pathlib.Path
                Dir containing files where GO terms are associated with a model

            Returns
            -------
            nx.MultiDiGraph
                The model-GO term bipartite DAG
        """
        dag = nx.MultiDiGraph()
        model_go_dict: Dict[str, List[str]] = {}
        go_model_dict: Dict[str, Set[str]] = defaultdict(set)
        for model_go_map_file in model_go_map_dir.iterdir():
            go_terms = parse_model_go_map(model_go_map_file)
            model_name = model_go_map_file.stem
            if model_name in model_go_dict:
                raise ValueError("Duplicate models in model_go_map_dir")
            model_go_dict[model_name] = go_terms
            for go_term in go_terms:
                for go_parent in self.go_dag.predecessors(go_term):
                    go_model_dict[go_parent].add(model_name)
        # Add nodes to the dag
        dag.add_nodes_from(list(model_go_dict.keys()), bipartite=0, model=None)
        dag.add_nodes_from(self.go_dag.nodes, bipartite=1)
        # Add edges to the dag
        for model_name in model_go_dict:
            dag.nodes[model_name]["label_vector"] = model_go_dict[model_name]
            for go_term in model_go_dict[model_name]:
                dag.add_edge(model_name, go_term)
        for go_term in go_model_dict:
            for model_name in go_model_dict[go_term]:
                dag.add_edge(go_term, model_name)
        return dag

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
