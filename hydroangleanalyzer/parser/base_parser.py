from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

import numpy as np


class BaseParser(ABC):
    """Abstract interface for trajectory parsers consumed by analyzers.

    Subclasses must implement frame parsing and frame count methods. Optional
    geometry helpers (box size, cylindrical conversion) can be overridden where
    supported by underlying file format.

    Parameters
    ----------
    in_path : str
        Path to trajectory / structure file.
    particle_type_wall : Any
        Identifier(s) for wall particles (type IDs for LAMMPS dump, symbols for
        ASE/XYZ).
    """

    @abstractmethod
    def __init__(self, in_path: str, particle_type_wall: Any):
        pass

    @abstractmethod
    def parse(self, num_frame: int, indices: Optional[np.ndarray] = None) -> np.ndarray:
        """Return Cartesian coordinates for selected atoms in a frame.

        Parameters
        ----------
        num_frame : int
            Frame index.
        indices : ndarray[int], optional
            Atom indices to select; if None return all atoms.

        Returns
        -------
        ndarray, shape (M, 3)
            Particle coordinates.
        """
        pass

    @abstractmethod
    def frame_tot(self) -> int:
        """Return total number of frames available.

        Returns
        -------
        int
            Number of frames.
        """
        pass

    def box_size_x(self, num_frame: int) -> float:  # pragma: no cover - default
        """Return box x-length for frame (override if available)."""
        raise NotImplementedError("box_size_x not implemented for this parser.")

    def box_size_y(self, num_frame: int) -> float:  # pragma: no cover - default
        """Return box y-length for frame (override if available)."""
        raise NotImplementedError("box_size_y not implemented for this parser.")

    def box_length_max(self, num_frame: int) -> float:  # pragma: no cover - default
        """Return maximum box length for frame (override if available)."""
        raise NotImplementedError("box_length_max not implemented for this parser.")

    def return_cylindrical_coord_pars(
        self,
        frame_list: List[int],
        type_model: str = "cylinder_y",
        liquid_indices: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, int]:  # pragma: no cover - default
        """Return cylindrical coordinate arrays for frames (override if available)."""
        raise NotImplementedError(
            "return_cylindrical_coord_pars not implemented for this parser."
        )
