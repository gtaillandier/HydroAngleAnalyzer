from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Tuple

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
    def parse(self, frame_index: int, indices: np.ndarray | None = None) -> np.ndarray:
        """Return Cartesian coordinates for selected atoms in a frame.

        Parameters
        ----------
        frame_index : int
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
    def frame_count(self) -> int:
        """Return the total number of frames available.

        Returns
        -------
        int
            Number of frames.
        """
        pass

    def frame_tot(self) -> int:
        """Return the total number of frames available. (Legacy name)."""
        import warnings

        warnings.warn(
            "frame_tot is deprecated, use frame_count instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.frame_count()

    def box_size_x(self, frame_index: int) -> float:  # pragma: no cover - default
        """Return the box x-length for a frame. (override if available)."""
        raise NotImplementedError("box_size_x not implemented for this parser.")

    def box_size_y(self, frame_index: int) -> float:  # pragma: no cover - default
        """Return the box y-length for a frame. (override if available)."""
        raise NotImplementedError("box_size_y not implemented for this parser.")

    def box_length_max(self, frame_index: int) -> float:  # pragma: no cover - default
        """Return the maximum box length for a frame. (override if available)."""
        raise NotImplementedError("box_length_max not implemented for this parser.")

    def get_cylindrical_coordinates(
        self,
        frame_list: List[int],
        type_model: str = "cylinder_y",
        liquid_indices: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, int]:  # pragma: no cover - default
        """Return cylindrical coordinate arrays for frames (override if available)."""
        raise NotImplementedError(
            "get_cylindrical_coordinates not implemented for this parser."
        )

    def return_cylindrical_coord_pars(self, *args, **kwargs):
        """Return cylindrical coordinate arrays for frames. (Legacy name)."""
        import warnings

        warnings.warn(
            "return_cylindrical_coord_pars is deprecated, "
            "use get_cylindrical_coordinates instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_cylindrical_coordinates(*args, **kwargs)
