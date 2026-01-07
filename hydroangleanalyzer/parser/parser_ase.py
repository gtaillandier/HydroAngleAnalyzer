from typing import List

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
                "The 'ase' package is required to use Ase_Parser. Install with "
                "'pip install ase'."
            ) from e
        self.in_path = in_path
        self.trajectory = read(self.in_path, index=":")

    def parse(self, num_frame, indices=None):
        """Return Cartesian coordinates for selected atoms in a frame.

        Parameters
        ----------
        num_frame : int
            Frame index.
        indices : sequence[int], optional
            Atom indices to select; if None all atoms are returned.

        Returns
        -------
        ndarray, shape (M, 3)
            Cartesian coordinates of requested atoms.
        """
        frame = self.trajectory[num_frame]
        if indices is not None:
            indices = np.array(indices)
            return frame.positions[indices]
        return frame.positions

    def parse_liquid(self, particle_type_liquid, num_frame):
        """Return liquid atom coordinates filtering by atomic symbol list.

        Parameters
        ----------
        particle_type_liquid : sequence[str]
            Symbols identifying liquid particles.
        num_frame : int
            Frame index.

        Returns
        -------
        ndarray, shape (L, 3)
            Liquid atom positions.
        """
        frame = self.trajectory[num_frame]
        mask = np.isin(frame.symbols, particle_type_liquid)
        return frame.positions[mask]

    def return_cylindrical_coord_pars(
        self, frame_list, type_model="cylinder_y", liquid_indices=None
    ):
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
        xi_par = np.array([])
        zi_par = np.array([])
        for frame_idx in frame_list:
            frame = self.trajectory[frame_idx]
            X_par = frame.positions
            if liquid_indices is not None:
                liquid_indices = np.array(liquid_indices)
                X_par = X_par[liquid_indices]
            X_cm = np.mean(X_par, axis=0)
            X_0 = X_par - X_cm
            X_0[:, 2] = X_par[:, 2]
            if type_model == "cylinder_y":
                xi_frame = np.abs(X_0[:, 0] + 0.01)
            elif type_model == "cylinder_x":
                xi_frame = np.abs(X_0[:, 1] + 0.01)
            else:
                xi_frame = np.sqrt(X_0[:, 0] ** 2 + X_0[:, 1] ** 2)
            zi_frame = X_0[:, 2]
            xi_par = np.concatenate((xi_par, xi_frame))
            zi_par = np.concatenate((zi_par, zi_frame))
            if frame_idx % 10 == 0:
                print(f"Frame: {frame_idx}\nCenter of Mass: {X_cm}")
        print(f"\nxi range:\t({np.min(xi_par)},{np.max(xi_par)})")
        print(f"zi range:\t({np.min(zi_par)},{np.max(zi_par)})")
        return xi_par, zi_par, len(frame_list)

    def box_size_y(self, num_frame):
        """Return y-dimension (a2y) of simulation cell for frame."""
        frame = self.trajectory[num_frame]
        return float(frame.cell[1, 1])

    def box_size_x(self, num_frame):
        """Return x-dimension (a1x) of simulation cell for frame."""
        frame = self.trajectory[num_frame]
        return float(frame.cell[0, 0])

    def box_lenght_max(self, num_frame):  # legacy spelling retained
        """Return maximum lattice vector length for frame (legacy name)."""
        frame = self.trajectory[num_frame]
        return float(max(frame.cell.lengths()))

    def frame_tot(self):
        """Return total number of frames in trajectory."""
        return len(self.trajectory)


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
                "The 'ase' package is required to use ASE_WaterMoleculeFinder. "
                "Install it with: pip install ase"
            ) from e
        self._ase_read = read
        self._NeighborList = NeighborList
        self.trajectory = self._ase_read(in_path, index=":")
        self.particle_type_wall = particle_type_wall
        self.oxygen_type = oxygen_type
        self.hydrogen_type = hydrogen_type
        self.oh_cutoff = oh_cutoff

    def get_water_oxygen_indices(self, num_frame):
        """Return indices of oxygen atoms each bonded to exactly two hydrogens.

        Parameters
        ----------
        num_frame : int
            Frame index.

        Returns
        -------
        ndarray
            Oxygen atom indices satisfying bonding criterion.
        """
        frame = self.trajectory[num_frame]
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

    def get_water_oxygen_positions(self, num_frame: int) -> np.ndarray:
        """Return Cartesian positions of water oxygen atoms for a frame.

        Parameters
        ----------
        num_frame : int
            Frame index.

        Returns
        -------
        ndarray, shape (N, 3)
            Oxygen atom positions; may be empty if none match criteria.
        """
        indices = self.get_water_oxygen_indices(num_frame)
        frame = self.trajectory[num_frame]
        return frame.positions[indices]


class AseWallParser:
    """Parser extracting wall particle coordinates (excluding liquid types).

    Parameters
    ----------
    in_path : str
        Path to trajectory file.
    particule_liquid_type : sequence[str]
        Symbols representing liquid particles to exclude.
    """

    def __init__(self, in_path, particule_liquid_type):
        try:
            from ase.io import read
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "The 'ase' package is required to use Ase_wallParser. Install it "
                "with: pip install ase"
            ) from e
        self.in_path = in_path
        self.particule_liquid_type = particule_liquid_type
        self.trajectory = read(self.in_path, index=":")

    def parse(self, num_frame):
        """Return wall coordinates for supplied frame index."""
        frame = self.trajectory[num_frame]
        mask = ~np.isin(frame.get_chemical_symbols(), self.particule_liquid_type)
        return frame.positions[mask]

    def find_highest_wall_part(self, num_frame):
        """Return maximum z-coordinate among wall particles for frame."""
        X_wall = self.parse(num_frame)
        return float(np.max(X_wall[:, 2]))

    def return_cylindrical_coord_pars(self, frame_list, type_model="cylinder"):
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
            (xi_par, zi_par, n_frames).
        """
        xi_par = np.array([])
        zi_par = np.array([])
        for frame_idx in frame_list:
            frame = self.trajectory[frame_idx]
            X_par = frame.positions
            X_cm = np.mean(X_par, axis=0)
            X_0 = X_par - X_cm
            X_0[:, 2] = X_par[:, 2]
            if type_model == "cylinder":
                xi_frame = np.abs(X_0[:, 0] + 0.01)
            else:  # spherical
                xi_frame = np.sqrt(X_0[:, 0] ** 2 + X_0[:, 1] ** 2)
            zi_frame = X_0[:, 2]
            xi_par = np.concatenate((xi_par, xi_frame))
            zi_par = np.concatenate((zi_par, zi_frame))
            if frame_idx % 10 == 0:
                print(f"Frame: {frame_idx}\nCenter of Mass: {X_cm}")
        print(f"\nxi range:\t({np.min(xi_par)},{np.max(xi_par)})")
        print(f"zi range:\t({np.min(zi_par)},{np.max(zi_par)})")
        return xi_par, zi_par, len(frame_list)

    def box_size_y(self, num_frame):
        """Return y-dimension (a2y) of simulation cell for frame."""
        frame = self.trajectory[num_frame]
        return float(frame.cell[1, 1])

    def box_lenght_max(self, num_frame):  # legacy spelling retained
        """Return maximum lattice vector length for frame."""
        frame = self.trajectory[num_frame]
        return float(max(frame.cell.lengths()))

    def frame_tot(self):
        """Return total number of frames in trajectory."""
        return len(self.trajectory)


# Example usage (commented for library import safety):
# parser = Ase_Parser('traj.extxyz')
# coords = parser.parse(num_frame=0)
