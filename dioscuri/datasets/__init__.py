from dioscuri.registry import Registry
from dioscuri.datasets.dataset import *
from dioscuri.datasets.enroute_graph_wrapper import *

DATASET_REGISTRY = Registry('DATASET')
DATASET_REGISTRY.register(ODAirportDataset)
DATASET_REGISTRY.register(GroundTruthDataset)
DATASET_REGISTRY.register(Lido21EnrouteAirwayDataset)