from dioscuri.registry import Registry
from dioscuri.criterion.distance import *

CRITERION_REGISTRY = Registry('CRITERION')
CRITERION_REGISTRY.register(GeopyGreatCircleDistance)