import numpy as np
from .base import Engine

class NumpyEngine(Engine):
    def array(self, x):
        return np.array(x, dtype=np.float64)

    def randn(self, shape):
        return np.random.randn(*shape)

    def to_device(self, x):
        return x

    def to_host(self, x):
        return x
