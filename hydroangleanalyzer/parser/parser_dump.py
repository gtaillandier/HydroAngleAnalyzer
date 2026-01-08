from __future__ import annotations

import logging
import warnings
from typing import List, Tuple

import numpy as np

from .base_parser import BaseParser

logger = logging.getLogger(__name__)


class DumpParser(BaseParser):
    def __init__(self, in_path: str):
        """Initialize LAMMPS dump parser via OVITO pipeline."""
        try:
            from ovito.io import import_file
            from ovito.modifiers import ComputePropertyModifier
        except ImportError as e:
            raise ImportError(
                "The 'ovito' package is required for DumpParser. Install with: "
                "pip install HydroAngleAnalyzer[ovito]"
            ) from e

        self.in_path = in_path
        self.pipeline = import_file(self.in_path)
        self.pipeline.modifiers.append(
            ComputePropertyModifier(expressions=["1"], output_property="Unity")
        )
        self.num_frames = self.pipeline.source.num_frames

    def parse(self, frame_index: int, indices: np.ndarray | None = None) -> np.ndarray:
        """Compute and return particle positions for a single frame.

        Parameters
        ----------
        frame_index : int
            Frame index.
        indices : ndarray, optional
            Atom indices to select; if None return all atoms.

        Returns
        -------
        ndarray, shape (M, 3)
            Particle coordinates.
        """
        data = self.pipeline.compute(frame_index)
        x_par = np.asarray(data.particles["Position"])
        particle_ids = np.asarray(data.particles["Particle Identifier"])
        if indices is not None:
            mask = np.isin(particle_ids, indices)
            x_par = x_par[mask]
        return x_par

    def get_cylindrical_coordinates(
        self, frame_list: list, type_model: str = "cylinder_x", liquid_indices=None
    ):
        """Convert Cartesian coordinates to cylindrical coordinates for frames."""
        xi_par = np.array([])
        zi_par = np.array([])
        for frame in frame_list:
            X_par = self.parse(frame, liquid_indices)
            dim = len(X_par[0, :])
            X_cm = [(X_par[:, i]).sum() / len(X_par[:, i]) for i in range(dim)]
            X_0 = [X_par[:, i] - X_cm[i] * (i < 2) for i in range(dim)]
            if type_model == "cylinder_y":
                xi_par_frame = np.abs(X_0[0] + 0.01)
            elif type_model == "cylinder_x":
                xi_par_frame = np.abs(X_0[1] + 0.01)
            else:  # spherical
                xi_par_frame = np.sqrt(X_0[0] ** 2 + X_0[1] ** 2)
            zi_par_frame = X_0[2]
            xi_par = np.concatenate((xi_par, xi_par_frame))
            zi_par = np.concatenate((zi_par, zi_par_frame))
            if frame % 10 == 0:
                print(f"Frame: {frame}")
                print(X_cm)
        print(f"\nxi range:\t({np.min(xi_par)},{np.max(xi_par)})")
        print(f"zi range:\t({np.min(zi_par)},{np.max(zi_par)})")
        return xi_par, zi_par, len(frame_list)

    def box_size_y(self, frame_index: int) -> float:
        """Return y-dimension of simulation box."""
        data = self.pipeline.compute(frame_index)
        y_vector = data.cell.matrix[1, :3]
        return float(np.linalg.norm(y_vector))

    def box_size_x(self, frame_index: int) -> float:
        """Return x-dimension of simulation box."""
        data = self.pipeline.compute(frame_index)
        x_vector = data.cell.matrix[0, :3]
        return float(np.linalg.norm(x_vector))

    def box_length_max(self, frame_index: int) -> float:
        """Return the maximum dimension of the simulation box.

        Parameters
        ----------
        frame_index : int
            Frame index.

        Returns
        -------
        float
            Maximum box length.
        """
        data = self.pipeline.compute(frame_index)
        y_vector = np.linalg.norm(data.cell.matrix[1, :3])
        x_vector = np.linalg.norm(data.cell.matrix[0, :3])
        z_vector = np.linalg.norm(data.cell.matrix[2, :3])
        return float(np.max(np.array([y_vector, x_vector, z_vector])))

    def frame_count(self) -> int:
        """Return the total number of frames in the trajectory.

        Returns
        -------
        int
            Number of frames.
        """
        return self.num_frames


class DumpWallParser:
    """Parser for extracting wall particle coordinates from LAMMPS dump files.

    Parameters
    ----------
    in_path : str
        Path to LAMMPS dump file.
    liquid_particle_types : List[int]
        Type IDs of particles to exclude as liquid.
    """

    def __init__(self, in_path: str, liquid_particle_types: List[int]):
        self.in_path = in_path
        self.liquid_particle_types = liquid_particle_types
        self.pipeline = self.load_dump_ovito()

    def load_dump_ovito(self):
        try:
            from ovito.io import import_file
            from ovito.modifiers import (
                ComputePropertyModifier,
                DeleteSelectedModifier,
                SelectTypeModifier,
            )
        except ImportError as e:
            raise ImportError(
                "OVITO required. Install with: pip install HydroAngleAnalyzer[ovito]"
            ) from e
        pipeline = import_file(self.in_path)
        pipeline.modifiers.append(
            SelectTypeModifier(
                property="Particle Type", types=self.liquid_particle_types
            )
        )
        pipeline.modifiers.append(DeleteSelectedModifier())
        pipeline.modifiers.append(
            ComputePropertyModifier(expressions=["1"], output_property="Unity")
        )
        return pipeline

    def parse(self, frame_index):
        data = self.pipeline.compute(frame_index)
        return np.asarray(data.particles["Position"])

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
        data = self.pipeline.compute(frame_index)
        x_wall = np.asarray(data.particles["Position"])
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
        """Convert Cartesian coordinates to cylindrical coordinates for frames.

        Parameters
        ----------
        frame_list : sequence[int]
            Frames to process.
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
            data = self.pipeline.compute(frame_idx)
            x_par = np.asarray(data.particles["Position"])
            dim = x_par.shape[1]
            x_cm = [(x_par[:, i]).sum() / len(x_par[:, i]) for i in range(dim)]
            x_0 = [x_par[:, i] - x_cm[i] * (i < 2) for i in range(dim)]
            if type_model == "cylinder":
                xi_par_frame = np.abs(x_0[0] + 0.01)
            else:  # spherical
                xi_par_frame = np.sqrt(x_0[0] ** 2 + x_0[1] ** 2)
            zi_par_frame = x_0[2]
            xi_values = np.concatenate((xi_values, xi_par_frame))
            zi_values = np.concatenate((zi_values, zi_par_frame))
            if frame_idx % 10 == 0:
                print(f"frame: {frame_idx}")
                print(x_cm)
        print(f"\nxi range:\t({np.min(xi_values)},{np.max(xi_values)})")
        print(f"zi range:\t({np.min(zi_values)},{np.max(zi_values)})")
        return xi_values, zi_values, len(frame_list)

    def box_size_y(self, frame_index: int) -> float:
        """Return the y-dimension of the simulation box."""
        data = self.pipeline.compute(frame_index)
        y_vector = data.cell.matrix[1, :3]
        return float(np.linalg.norm(y_vector))

    def box_length_max(self, frame_index):  # legacy name kept
        data = self.pipeline.compute(frame_index)
        y_vector = np.linalg.norm(data.cell.matrix[1, :3])
        x_vector = np.linalg.norm(data.cell.matrix[0, :3])
        z_vector = np.linalg.norm(data.cell.matrix[2, :3])
        return np.max(np.array([y_vector, x_vector, z_vector]))

    def frame_count(self) -> int:
        """Return total number of frames."""
        return self.pipeline.source.num_frames


class DumpWaterMoleculeFinder:
    """Identify water oxygen atoms in a parsed LAMMPS trajectory."""

    def __init__(
        self,
        in_path: str,
        particle_type_wall: set,
        oxygen_type: int = 3,
        hydrogen_type: int = 2,
        oh_cutoff: float = 1.2,
    ):
        """Initialize water molecule finder with OVITO pipeline."""
        self.in_path = in_path
        self.particle_type_wall = particle_type_wall
        self.oxygen_type = oxygen_type
        self.hydrogen_type = hydrogen_type
        self.oh_cutoff = oh_cutoff
        self.pipeline = self._setup_pipeline()

    def _setup_pipeline(self):
        try:
            from ovito.io import import_file
            from ovito.modifiers import (
                ComputePropertyModifier,
                CoordinationAnalysisModifier,
            )
        except ImportError as e:
            raise ImportError(
                "OVITO required for water detection. Install: pip install "
                "HydroAngleAnalyzer[ovito]"
            ) from e
        pipeline = import_file(self.in_path)
        pipeline.modifiers.append(
            CoordinationAnalysisModifier(cutoff=self.oh_cutoff, number_of_bins=200)
        )
        expr = f"(ParticleType == {self.oxygen_type}) && (Coordination == 2)"
        pipeline.modifiers.append(
            ComputePropertyModifier(expressions=[expr], output_property="IsWaterOxygen")
        )
        return pipeline

    def get_water_oxygen_ids(self, frame_index: int) -> np.ndarray:
        """Return IDs of oxygen atoms belonging to water molecules."""
        data = self.pipeline.compute(frame_index)
        if "IsWaterOxygen" in data.particles:
            mask = np.array(data.particles["IsWaterOxygen"].array) == 1
            oxygen_indices = np.where(mask)[0]
            oxygen_ids = data.particles["Particle Identifier"][oxygen_indices]
            return oxygen_ids
        return self._manual_identification(data)


Dump_WaterMoleculeFinder = DumpWaterMoleculeFinder
DumpParse_wall = DumpWallParser
