from dioscuri.registry import Registry
from dioscuri.metrics.hausdorff import *

METRIC_REGISTRY = Registry('METRIC')
METRIC_REGISTRY.register(Hausdorff)