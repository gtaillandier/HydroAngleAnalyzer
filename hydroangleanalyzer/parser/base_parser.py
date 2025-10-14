from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Any, List, Set, Tuple

class BaseParser(ABC):
    """
    Abstract base class for trajectory parsers.
    All parsers must implement these methods.
    """

    @abstractmethod
    def __init__(self, in_path: str, particle_type_wall: Any):
        """
        Initialize the parser with the input file path and wall particle identifier(s).

        Args:
            in_path (str): Path to the input file.
            particle_type_wall (Any): Identifier(s) for wall particles (type numbers for LAMMPS, symbols for ASE/XYZ).
        """
        pass

    @abstractmethod
    def parse(self, num_frame: int, indices: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Parse the frame and return particle positions.

        Args:
            num_frame (int): The frame number to parse.
            indices (Optional[np.ndarray]): Array of particle indices to extract. If None, all particles are returned.

        Returns:
            np.ndarray: Positions of particles for the specified frame and indices.
        """
        pass

    @abstractmethod
    def frame_tot(self) -> int:
        """
        Return the total number of frames in the trajectory.

        Returns:
            int: Total number of frames.
        """
        pass

    def box_size_x(self, num_frame: int) -> float:
        """
        Return the x-dimension of the simulation box for a specific frame.
        Returns NaN if not implemented.
        """
        raise NotImplementedError("box_size_x not implemented for this parser.")

    def box_size_y(self, num_frame: int) -> float:
        """
        Return the y-dimension of the simulation box for a specific frame.
        Returns NaN if not implemented.
        """
        raise NotImplementedError("box_size_y not implemented for this parser.")

    def box_length_max(self, num_frame: int) -> float:
        """
        Return the maximum box dimension for a specific frame.
        Returns NaN if not implemented.
        """
        raise NotImplementedError("box_length_max not implemented for this parser.")

    def return_cylindrical_coord_pars(
        self,
        frame_list: List[int],
        type_model: str = "masspain_y",
        liquid_indices: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Convert Cartesian coordinates to cylindrical coordinates for the given frames and indices.
        Returns NaN if not implemented.
        """
        raise NotImplementedError("return_cylindrical_coord_pars not implemented for this parser.")