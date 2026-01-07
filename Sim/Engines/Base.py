from abc import ABC, abstractmethod

class Engine(ABC):
    @abstractmethod
    def array(self, x):
        """Convert input to engine-specific array."""
        pass

    @abstractmethod
    def randn(self, shape):
        """Return normally distributed noise."""
        pass

    @abstractmethod
    def to_device(self, x):
        """Move array to engine device (GPU/CPU)."""
        pass

    @abstractmethod
    def to_host(self, x):
        """Move array back to host (CPU)."""
        pass
