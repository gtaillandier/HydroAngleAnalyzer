from __future__ import annotations

import warnings
from typing import List, Tuple

import numpy as np

from .base_parser import BaseParser


class AseParser(BaseParser):
    """ASE-backed trajectory parser exposing minimal interface for analyzers.

    Parameters
    ----------
    in_path : str
        Path to any ASE-readable trajectory/file pattern (e.g. XYZ, extxyz,
        POSCAR, etc.).
    """

    def __init__(self, in_path: str) -> None:
        try:
            from ase.io import read
        except ImportError as e:  # pragma: no cover - dependency guard
            raise ImportError(
                "The 'ase' package is required to use AseParser. Install with "
                "'pip install ase'."
            ) from e
        self.in_path = in_path
        self.trajectory = read(self.in_path, index=":")

    def parse(self, frame_index: int, indices: np.ndarray | None = None) -> np.ndarray:
        """Return Cartesian coordinates for selected atoms in a frame.

        Parameters
        ----------
        frame_index : int
            Frame index.
        indices : sequence[int], optional
            Atom indices to select; if None all atoms are returned.

        Returns
        -------
        ndarray, shape (M, 3)
            Cartesian coordinates of requested atoms.
        """
        frame = self.trajectory[frame_index]
        if indices is not None:
            indices = np.array(indices)
            return frame.positions[indices]
        return frame.positions

    def parse_liquid_particles(
        self, liquid_particle_types: List[str], frame_index: int
    ) -> np.ndarray:
        """Return liquid atom coordinates filtering by atomic symbol list.

        Parameters
        ----------
        liquid_particle_types : sequence[str]
            Symbols identifying liquid particles.
        frame_index : int
            Frame index.

        Returns
        -------
        ndarray, shape (L, 3)
            Liquid atom positions.
        """
        frame = self.trajectory[frame_index]
        mask = np.isin(frame.symbols, liquid_particle_types)
        return frame.positions[mask]

    def parse_liquid(self, *args, **kwargs):
        """Deprecated alias for parse_liquid_particles."""
        warnings.warn(
            "parse_liquid is deprecated, use parse_liquid_particles instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.parse_liquid_particles(*args, **kwargs)

    def get_cylindrical_coordinates(
        self,
        frame_list: List[int],
        type_model: str = "cylinder_y",
        liquid_indices: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Return cylindrical projection arrays for multiple frames.

        Parameters
        ----------
        frame_list : sequence[int]
            Frames to process.
        type_model : str, default "cylinder_y"
            One of {"cylinder_y", "cylinder_x", "spherical"}.
        liquid_indices : sequence[int], optional
            Subset indices of atoms to include.

        Returns
        -------
        tuple(ndarray, ndarray, int)
            (xi_values, zi_values, n_frames) flattened over frames.
        """
        xi_values = np.array([])
        zi_values = np.array([])
        for frame_idx in frame_list:
            frame = self.trajectory[frame_idx]
            x_par = frame.positions
            if liquid_indices is not None:
                liquid_indices = np.array(liquid_indices)
                x_par = x_par[liquid_indices]
            x_cm = np.mean(x_par, axis=0)
            x_0 = x_par - x_cm
            x_0[:, 2] = x_par[:, 2]
            if type_model == "cylinder_y":
                xi_frame = np.abs(x_0[:, 0] + 0.01)
            elif type_model == "cylinder_x":
                xi_frame = np.abs(x_0[:, 1] + 0.01)
            else:
                xi_frame = np.sqrt(x_0[:, 0] ** 2 + x_0[:, 1] ** 2)
            zi_frame = x_0[:, 2]
            xi_values = np.concatenate((xi_values, xi_frame))
            zi_values = np.concatenate((zi_values, zi_frame))
            if frame_idx % 10 == 0:
                print(f"Frame: {frame_idx}\nCenter of Mass: {x_cm}")
        print(f"\nxi range:\t({np.min(xi_values)},{np.max(xi_values)})")
        print(f"zi range:\t({np.min(zi_values)},{np.max(zi_values)})")
        return xi_values, zi_values, len(frame_list)

    def return_cylindrical_coord_pars(self, *args, **kwargs):
        """Deprecated alias for get_cylindrical_coordinates."""
        warnings.warn(
            "return_cylindrical_coord_pars is deprecated, "
            "use get_cylindrical_coordinates instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_cylindrical_coordinates(*args, **kwargs)

    def box_size_y(self, frame_index: int) -> float:
        """Return y-dimension (a2y) of simulation cell for frame."""
        frame = self.trajectory[frame_index]
        return float(frame.cell[1, 1])

    def box_size_x(self, frame_index: int) -> float:
        """Return x-dimension (a1x) of simulation cell for frame."""
        frame = self.trajectory[frame_index]
        return float(frame.cell[0, 0])

    def box_length_max(self, frame_index: int) -> float:
        """Return maximum lattice vector length for frame."""
        frame = self.trajectory[frame_index]
        return float(max(frame.cell.lengths()))

    def frame_count(self) -> int:
        """Return total number of frames in trajectory."""
        return len(self.trajectory)

    def frame_tot(self) -> int:
        """Deprecated alias for frame_count."""
        warnings.warn(
            "frame_tot is deprecated, use frame_count instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.frame_count()


class AseWaterMoleculeFinder:
    """Identify water oxygen atoms by counting hydrogen neighbors.

    Uses ASE neighbor list to find oxygens with exactly two hydrogens.

    Parameters
    ----------
    in_path : str
        Path to ASE-readable trajectory.
    particle_type_wall : sequence[str]
        Symbols representing wall particles (unused presently, reserved for
        filtering).
    oxygen_type : str, default "O"
        Oxygen atom symbol.
    hydrogen_type : str, default "H"
        Hydrogen atom symbol.
    oh_cutoff : float, default 1.2
        O–H distance cutoff used to detect bonded hydrogens.
    """

    def __init__(
        self,
        in_path: str,
        particle_type_wall: List[str],
        oxygen_type: str = "O",
        hydrogen_type: str = "H",
        oh_cutoff: float = 1.2,
    ):
        try:
            from ase.io import read
            from ase.neighborlist import NeighborList
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "The 'ase' package is required to use AseWaterMoleculeFinder. "
                "Install it with: pip install ase"
            ) from e
        self._ase_read = read
        self._NeighborList = NeighborList
        self.trajectory = self._ase_read(in_path, index=":")
        self.particle_type_wall = particle_type_wall
        self.oxygen_type = oxygen_type
        self.hydrogen_type = hydrogen_type
        self.oh_cutoff = oh_cutoff

    def get_water_oxygen_indices(self, frame_index: int) -> np.ndarray:
        """Return indices of oxygen atoms each bonded to exactly two hydrogens.

        Parameters
        ----------
        frame_index : int
            Frame index.

        Returns
        -------
        ndarray
            Oxygen atom indices satisfying bonding criterion.
        """
        frame = self.trajectory[frame_index]
        symbols = np.array(frame.get_chemical_symbols())
        oxygen_indices = np.where(symbols == self.oxygen_type)[0]
        hydrogen_indices = np.where(symbols == self.hydrogen_type)[0]
        cutoffs = [self.oh_cutoff] * len(frame)
        nl = self._NeighborList(cutoffs, self_interaction=False, bothways=True)
        nl.update(frame)
        water_oxygens = []
        for o_idx in oxygen_indices:
            indices, _offsets = nl.get_neighbors(o_idx)
            h_neighbors = [i for i in indices if i in hydrogen_indices]
            if len(h_neighbors) == 2:
                water_oxygens.append(o_idx)
        return np.array(water_oxygens, dtype=int)

    def get_water_oxygen_positions(self, frame_index: int) -> np.ndarray:
        """Return Cartesian positions of water oxygen atoms for a frame.

        Parameters
        ----------
        frame_index : int
            Frame index.

        Returns
        -------
        ndarray, shape (N, 3)
            Oxygen atom positions; may be empty if none match criteria.
        """
        indices = self.get_water_oxygen_indices(frame_index)
        frame = self.trajectory[frame_index]
        return frame.positions[indices]


class AseWallParser:
    """Parser extracting wall particle coordinates (excluding liquid types).

    Parameters
    ----------
    in_path : str
        Path to trajectory file.
    liquid_particle_types : sequence[str]
        Symbols representing liquid particles to exclude.
    """

    def __init__(self, in_path: str, liquid_particle_types: List[str]):
        try:
            from ase.io import read
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "The 'ase' package is required to use AseWallParser. Install it "
                "with: pip install ase"
            ) from e
        self.in_path = in_path
        self.liquid_particle_types = liquid_particle_types
        self.trajectory = read(self.in_path, index=":")

    def parse(self, frame_index: int) -> np.ndarray:
        """Return wall coordinates for the supplied frame index.

        Parameters
        ----------
        frame_index : int
            Frame index.

        Returns
        -------
        ndarray
            Wall particle coordinates.
        """
        frame = self.trajectory[frame_index]
        mask = ~np.isin(frame.get_chemical_symbols(), self.liquid_particle_types)
        return frame.positions[mask]

    def find_highest_wall_particle(self, frame_index: int) -> float:
        """Return the maximum z-coordinate among wall particles for a frame.

        Parameters
        ----------
        frame_index : int
            Frame index.

        Returns
        -------
        float
            Maximum z-coordinate.
        """
        x_wall = self.parse(frame_index)
        return float(np.max(x_wall[:, 2]))

    def find_highest_wall_part(self, *args, **kwargs):
        """Deprecated alias for find_highest_wall_particle."""
        warnings.warn(
            "find_highest_wall_part is deprecated, "
            "use find_highest_wall_particle instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.find_highest_wall_particle(*args, **kwargs)

    def get_cylindrical_coordinates(
        self,
        frame_list: List[int],
        type_model: str = "cylinder",
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Return cylindrical projections for wall particles across frames.

        Parameters
        ----------
        frame_list : sequence[int]
            Frame indices.
        type_model : str, default "cylinder"
            Either "cylinder" or "spherical".

        Returns
        -------
        tuple(ndarray, ndarray, int)
            (xi_values, zi_values, n_frames).
        """
        xi_values = np.array([])
        zi_values = np.array([])
        for frame_idx in frame_list:
            frame = self.trajectory[frame_idx]
            x_par = frame.positions
            x_cm = np.mean(x_par, axis=0)
            x_0 = x_par - x_cm
            x_0[:, 2] = x_par[:, 2]
            if type_model == "cylinder":
                xi_frame = np.abs(x_0[:, 0] + 0.01)
            else:  # spherical
                xi_frame = np.sqrt(x_0[:, 0] ** 2 + x_0[:, 1] ** 2)
            zi_frame = x_0[:, 2]
            xi_values = np.concatenate((xi_values, xi_frame))
            zi_values = np.concatenate((zi_values, zi_frame))
            if frame_idx % 10 == 0:
                print(f"Frame: {frame_idx}\nCenter of Mass: {x_cm}")
        print(f"\nxi range:\t({np.min(xi_values)},{np.max(xi_values)})")
        print(f"zi range:\t({np.min(zi_values)},{np.max(zi_values)})")
        return xi_values, zi_values, len(frame_list)

    def return_cylindrical_coord_pars(self, *args, **kwargs):
        """Deprecated alias for get_cylindrical_coordinates."""
        warnings.warn(
            "return_cylindrical_coord_pars is deprecated, "
            "use get_cylindrical_coordinates instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_cylindrical_coordinates(*args, **kwargs)

    def box_size_y(self, frame_index: int) -> float:
        """Return y-dimension (a2y) of simulation cell for frame."""
        frame = self.trajectory[frame_index]
        return float(frame.cell[1, 1])

    def box_length_max(self, frame_index: int) -> float:
        """Return maximum lattice vector length for frame."""
        frame = self.trajectory[frame_index]
        return float(max(frame.cell.lengths()))

    def frame_count(self) -> int:
        """Return total number of frames in trajectory."""
        return len(self.trajectory)

    def frame_tot(self) -> int:
        """Deprecated alias for frame_count."""
        warnings.warn(
            "frame_tot is deprecated, use frame_count instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.frame_count()


# Example usage (commented for library import safety):
# parser = Ase_Parser('traj.extxyz')
# coords = parser.parse(frame_indexs=0)

Ase_Parser = AseParser
Ase_WaterMoleculeFinder = AseWaterMoleculeFinder
Ase_WallParser = AseWallParser
