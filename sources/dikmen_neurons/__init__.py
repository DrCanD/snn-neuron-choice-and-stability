"""Study-required MS-IF subset of dikmen-spiking-neurons (Apache-2.0).
Modified packaging: register only the seven MS-IF models used in this study.
The neuron implementations in base.py and ms_if are unmodified.
"""
from .base import BaseNeuron, spike_hard, spike_stochastic
from .ms_if import REGISTRY
__version__ = "1.2.0"
class NeuronRegistry:
    _all = dict(REGISTRY)
    @classmethod
    def create(cls, name, size=128, **kwargs):
        return cls._all[name](size=size, **kwargs)
    @classmethod
    def list_all(cls):
        return sorted(cls._all)
