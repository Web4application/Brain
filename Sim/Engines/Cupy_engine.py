import cupy as cp
from .base import Engine

class CuPyEngine(Engine):
    def array(self, x):
        return cp.array(x, dtype=cp.float64)

    def randn(self, shape):
        return cp.random.randn(*shape)

    def to_device(self, x):
        return cp.array(x)

    def to_host(self, x):
        return cp.asnumpy(x)
