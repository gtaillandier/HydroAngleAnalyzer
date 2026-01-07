import logging

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

    def parse(self, num_frame: int, indices: np.ndarray = None) -> np.ndarray:
        """Compute and return particle positions for a single frame."""
        data = self.pipeline.compute(num_frame)
        X_par = np.asarray(data.particles["Position"])
        particle_ids = np.asarray(data.particles["Particle Identifier"])
        if indices is not None:
            mask = np.isin(particle_ids, indices)
            X_par = X_par[mask]
        return X_par

    def return_cylindrical_coord_pars(
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

    def box_size_y(self, num_frame: int) -> float:
        """Return y-dimension of simulation box."""
        data = self.pipeline.compute(num_frame)
        y_vector = data.cell.matrix[1, :3]
        return float(np.linalg.norm(y_vector))

    def box_size_x(self, num_frame: int) -> float:
        """Return x-dimension of simulation box."""
        data = self.pipeline.compute(num_frame)
        x_vector = data.cell.matrix[0, :3]
        return float(np.linalg.norm(x_vector))

    def box_length_max(self, num_frame: int) -> float:
        """Return maximum dimension of simulation box."""
        data = self.pipeline.compute(num_frame)
        y_vector = np.linalg.norm(data.cell.matrix[1, :3])
        x_vector = np.linalg.norm(data.cell.matrix[0, :3])
        z_vector = np.linalg.norm(data.cell.matrix[2, :3])
        return np.max(np.array([y_vector, x_vector, z_vector]))

    def frame_tot(self) -> int:
        """Return total number of frames."""
        return self.pipeline.source.num_frames


class DumpWallParser:
    def __init__(self, in_path, particule_liquid_type):
        self.in_path = in_path
        self.particule_liquid_type = particule_liquid_type
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
                property="Particle Type", types=self.particule_liquid_type
            )
        )
        pipeline.modifiers.append(DeleteSelectedModifier())
        pipeline.modifiers.append(
            ComputePropertyModifier(expressions=["1"], output_property="Unity")
        )
        return pipeline

    def parse(self, num_frame):
        data = self.pipeline.compute(num_frame)
        return np.asarray(data.particles["Position"])

    def find_highest_wall_part(self, num_frame):
        data = self.pipeline.compute(num_frame)
        X_wall = np.asarray(data.particles["Position"])
        return np.max(X_wall[:, 2])

    def return_cylindrical_coord_pars(self, frame_list, type_model="cylinder"):
        """Convert Cartesian coordinates to cylindrical coordinates for frames."""
        xi_par = np.array([])
        zi_par = np.array([])
        for frame in frame_list:
            data = self.pipeline.compute(frame)
            X_par = np.asarray(data.particles["Position"])
            dim = len(X_par[0, :])
            X_cm = [(X_par[:, i]).sum() / len(X_par[:, i]) for i in range(dim)]
            X_0 = [X_par[:, i] - X_cm[i] * (i < 2) for i in range(dim)]
            if type_model == "cylinder":
                xi_par_frame = np.abs(X_0[0] + 0.01)
            else:  # spherical
                xi_par_frame = np.sqrt(X_0[0] ** 2 + X_0[1] ** 2)
            zi_par_frame = X_0[2]
            xi_par = np.concatenate((xi_par, xi_par_frame))
            zi_par = np.concatenate((zi_par, zi_par_frame))
            if frame % 10 == 0:
                print(f"frame: {frame}")
                print(X_cm)
        print(f"\nxi range:\t({np.min(xi_par)},{np.max(xi_par)})")
        print(f"zi range:\t({np.min(zi_par)},{np.max(zi_par)})")
        return xi_par, zi_par, len(frame_list)

    def box_size_y(self, num_frame):
        data = self.pipeline.compute(num_frame)
        y_vector = data.cell.matrix[1, :3]
        return float(np.linalg.norm(y_vector))

    def box_lenght_max(self, num_frame):  # legacy name kept
        data = self.pipeline.compute(num_frame)
        y_vector = np.linalg.norm(data.cell.matrix[1, :3])
        x_vector = np.linalg.norm(data.cell.matrix[0, :3])
        z_vector = np.linalg.norm(data.cell.matrix[2, :3])
        return np.max(np.array([y_vector, x_vector, z_vector]))

    def frame_tot(self):
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

    def get_water_oxygen_ids(self, num_frame: int) -> np.ndarray:
        """Return IDs of oxygen atoms belonging to water molecules."""
        data = self.pipeline.compute(num_frame)
        if "IsWaterOxygen" in data.particles:
            mask = np.array(data.particles["IsWaterOxygen"].array) == 1
            oxygen_indices = np.where(mask)[0]
            oxygen_ids = data.particles["Particle Identifier"][oxygen_indices]
            return oxygen_ids
        return self._manual_identification(data)


Dump_WaterMoleculeFinder = DumpWaterMoleculeFinder
DumpParse_wall = DumpWallParser
