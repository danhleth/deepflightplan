from dioscuri.registry import Registry
from dioscuri.metrics.distance import *

METRIC_REGISTRY = Registry('METRIC')
METRIC_REGISTRY.register(HausdorffDistance)
METRIC_REGISTRY.register(DiffTotalDistance)