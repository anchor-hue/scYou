"""
scYOU: Single-Cell YOU model for proteomic data analysis
"""
from .models import CellEncoder, CellDecoder, ProteinEncoder, ProteinDecoder, ClusterLayer, DualPathAE
from .trainer import scYOU
from .utils import set_seed, create_data_object, myload_sc_proteomic_features, select_top_var_proteins, load_go_similarity_matrix, evaluate_clustering

__version__ = "1.0.0"
__author__ = "Research Team"