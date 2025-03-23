from dioscuri.registry import Registry

from dioscuri.algorithms.nx_astar import *
from dioscuri.algorithms.nx_yen import *
ALGORITHM_REGISTRY = Registry('ALGORITHM')

ALGORITHM_REGISTRY.register(nxAstar)
ALGORITHM_REGISTRY.register(nxYen)