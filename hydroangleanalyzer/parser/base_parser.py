from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Optional, List, Union, Set, Any

class BaseParser(ABC):
    """Abstract base class for trajectory parsers."""

    @abstractmethod
    def __init__(self, in_path: str, wall_identifier: Any):
        """
        Initialize the parser with the input file path and wall particle identifier.

        Args:
            in_path: Path to the input file.
            wall_identifier: Identifier for wall particles (element symbols for XYZ/ASE, type numbers for LAMMPS).
        """
        self.in_path = in_path
        self.wall_identifier = wall_identifier

    @abstractmethod
    def parse(self, num_frame: int, indices: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Parse the frame and return particle positions.

        Args:
            num_frame: The frame number to parse.
            indices: Array of particle indices to extract. If None, all particles are returned.

        Returns:
            np.ndarray: Positions of particles for the specified frame and indices.
        """
        pass

    @abstractmethod
    def frame_tot(self) -> int:
        """Return the total number of frames in the trajectory."""
        pass