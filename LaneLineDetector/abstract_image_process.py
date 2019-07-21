"""Provide a entity for abstract image processor."""

from abc import ABC, abstractmethod
import numpy as np


class AbstractImageProcessor(ABC):
    """Representation of an abstract image processor."""

    @abstractmethod
    def process(self) -> np.ndarray:
        """Perform image processing."""
        pass
