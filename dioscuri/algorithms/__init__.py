from dioscuri.registry import Registry

from dioscuri.algorithms.nx_astar import *
ALGORITHM_REGISTRY = Registry('ALGORITHM')

ALGORITHM_REGISTRY.register(nxAstar)