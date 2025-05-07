from dioscuri.registry import Registry
from dioscuri.distance.gcd_distance import *

DISTANCE_REGISTRY = Registry('DISTANCE')
DISTANCE_REGISTRY.register(GeopyGreatCircleDistance)