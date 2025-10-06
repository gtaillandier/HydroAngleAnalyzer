from ovito.io import import_file, export_file
from ovito.modifiers import (SelectTypeModifier, DeleteSelectedModifier, ComputePropertyModifier)

import numpy as np
from ovito.io import import_file, export_file
from ovito.modifiers import (SelectTypeModifier, DeleteSelectedModifier, ComputePropertyModifier,CoordinationAnalysisModifier )
from .base_parser import BaseParser
from typing import Union, List, Set, Optional, Any

import logging

logger = logging.getLogger(__name__)

class DumpParser(BaseParser):
    def __init__(self, in_path: str, particle_type_wall):
        """
        Initialize the parser for a LAMMPS dump file.
        This version does NOT preload all frames into memory.
        """
        self.in_path = in_path
        self.particle_type_wall = {int(particle) for particle in particle_type_wall}
        
        logger.info(f"[DumpParser] Importing file: {self.in_path}")
        self.pipeline = import_file(self.in_path)
        self.pipeline.modifiers.append(ComputePropertyModifier(expressions=['1'], output_property='Unity'))
        
        self.num_frames = self.pipeline.source.num_frames
        logger.info(f"[DumpParser] Initialized with {self.num_frames} frames.")

    def parse(self, num_frame: int, indices: np.ndarray = None) -> np.ndarray:
        """
        Compute and return particle positions for a single frame.
        """
        logger.info(f"[DumpParser] Computing frame {num_frame}")
        data = self.pipeline.compute(num_frame)
        X_par = np.asarray(data.particles["Position"])
        if indices is not None:
            X_par = X_par[indices]
        return X_par


    # Convert Cartesian coordinates to cylindrical coordinates
    def return_cylindrical_coord_pars(self, frame_list, type_model="masspain", liquid_indices=None):
        """Convert Cartesian coordinates to cylindrical coordinates for the given frames and indices."""
        xi_par = np.array([])
        zi_par = np.array([])

        for frame in frame_list:
            data = self.pipeline.compute(frame)
            X_par = np.asarray(data.particles["Position"])

            # Filter particles based on liquid_indices if provided
            if liquid_indices is not None:
                liquid_indices = np.array(liquid_indices)
                X_par = X_par[liquid_indices]

            dim = len(X_par[0, :])
            X_cm = [(X_par[:, i]).sum() / len(X_par[:, i]) for i in range(dim)]
            X_0 = [X_par[:, i] - X_cm[i] * (i < 2) for i in range(dim)]

            if type_model == "masspain":
                xi_par_frame = np.abs(X_0[0] + 0.01)
            elif type_model == "spherical":
                xi_par_frame = np.sqrt(X_0[0]**2 + X_0[1]**2)

            zi_par_frame = X_0[2]

            xi_par = np.concatenate((xi_par, xi_par_frame))
            zi_par = np.concatenate((zi_par, zi_par_frame))

            if frame % 10 == 0:
                print(f"frame: {frame}")
                print(X_cm)

        print("\nxi range:\t({},{})".format(np.min(xi_par), np.max(xi_par)))
        print("zi range:\t({},{})".format(np.min(zi_par), np.max(zi_par)))

        return xi_par, zi_par, len(frame_list)

    def box_size_y(self, num_frame):
        data = self.pipeline.compute(num_frame)
        y_vector = data.cell.matrix[1, :3]
        y_width = np.linalg.norm(y_vector)
        return y_width
    def box_size_x(self, num_frame):
        x_vector = data.cell.matrix[0, :3]  # Extract the x-vector from the cell matrix
        x_width = np.linalg.norm(x_vector)  # Calculate the magnitude of the x-vector
        return x_width
    def box_lenght_max(self, num_frame):
        data = self.pipeline.compute(num_frame)
        y_vector = np.linalg.norm(data.cell.matrix[1, :3])
        x_vector = np.linalg.norm(data.cell.matrix[0, :3])
        z_vector = np.linalg.norm(data.cell.matrix[2, :3])
        return np.max(np.array([y_vector,x_vector,z_vector]))
    def frame_tot(self):
        return self.pipeline.source.num_frames

class DumpParse_wall:
    def __init__(self, in_path,particule_liquid_type):
        self.in_path = in_path
        self.particule_liquid_type = particule_liquid_type
        self.pipeline = self.load_dump_ovito()

    def load_dump_ovito(self):
        # Load the file using OVITO
        pipeline = import_file(self.in_path)
        pipeline.modifiers.append(SelectTypeModifier(property='Particle Type', types=self.particule_liquid_type))
        pipeline.modifiers.append(DeleteSelectedModifier())
        pipeline.modifiers.append(ComputePropertyModifier(expressions=['1'], output_property='Unity'))
        return pipeline

    def parse(self, num_frame):
        data = self.pipeline.compute(num_frame)
        X_par = np.asarray(data.particles["Position"])
        return X_par

    def find_highest_wall_part(self, num_frame):
        data = self.pipeline.compute(num_frame)
        X_wall = np.asarray(data.particles["Position"])
        return np.max(X_wall[:, 2])

    # Convert Cartesian coordinates to cylindrical coordinates
    def return_cylindrical_coord_pars(self, frame_list, type_model="masspain"):
        """Convert Cartesian coordinates to cylindrical coordinates for the given frames."""
        xi_par = np.array([])
        zi_par = np.array([])
        for frame in frame_list:
            data = self.pipeline.compute(frame)
            X_par = np.asarray(data.particles["Position"])
            dim = len(X_par[0, :])
            X_cm = [(X_par[:, i]).sum() / len(X_par[:, i]) for i in range(dim)]
            X_0 = [X_par[:, i] - X_cm[i] * (i < 2) for i in range(dim)]
            if type_model == "masspain":
                xi_par_frame = np.abs(X_0[0]+ 0.01)
            elif type_model == "spherical":
                xi_par_frame = np.sqrt(X_0[0]**2 + X_0[1]**2)
            zi_par_frame = X_0[2]
            xi_par = np.concatenate((xi_par, xi_par_frame))
            zi_par = np.concatenate((zi_par, zi_par_frame))
            if frame % 10 == 0:
                print(f"frame: {frame}")
                print(X_cm)
        print("\nxi range:\t({},{})".format(np.min(xi_par), np.max(xi_par)))
        print("zi range:\t({},{})".format(np.min(zi_par), np.max(zi_par)))
        return xi_par, zi_par, len(frame_list)

    def box_size_y(self, num_frame):
        data = self.pipeline.compute(num_frame)
        y_vector = data.cell.matrix[1, :3]
        y_width = np.linalg.norm(y_vector)
        return y_width

    def box_lenght_max(self, num_frame):
        data = self.pipeline.compute(num_frame)
        y_vector = np.linalg.norm(data.cell.matrix[1, :3])
        x_vector = np.linalg.norm(data.cell.matrix[0, :3])
        z_vector = np.linalg.norm(data.cell.matrix[2, :3])
        return np.max(np.array([y_vector,x_vector,z_vector]))
    def frame_tot(self):
        return self.pipeline.source.num_frames

class Dump_WaterMoleculeFinder:
    """Identify water oxygen atoms in a parsed LAMMPS trajectory."""

    def __init__(
        self,
        in_path: str,
        particle_type_wall: set,
        oxygen_type: int = 3,
        hydrogen_type: int = 2,
        oh_cutoff: float = 1.2,
    ):
        """
        Initialize the water molecule finder.

        Args:
            in_path: Path to the input file.
            particle_type_wall: Set of particle types for wall particles.
            oxygen_type: Atom type ID of oxygen.
            hydrogen_type: Atom type ID of hydrogen.
            oh_cutoff: O–H distance cutoff for identifying water molecules.
        """
        self.in_path = in_path
        self.particle_type_wall = particle_type_wall
        self.oxygen_type = oxygen_type
        self.hydrogen_type = hydrogen_type
        self.oh_cutoff = oh_cutoff

        # Load the file and set up the OVITO pipeline
        self.pipeline = self._setup_pipeline()

    def _setup_pipeline(self):
        """Set up the OVITO pipeline with water molecule detection."""
        pipeline = import_file(self.in_path)

        # Add coordination analysis modifier
        pipeline.modifiers.append(
            CoordinationAnalysisModifier(cutoff=self.oh_cutoff, number_of_bins=200)
        )

        # Add compute property modifier to identify water oxygens
        expr = f"(ParticleType == {self.oxygen_type}) && (Coordination == 2)"
        pipeline.modifiers.append(
            ComputePropertyModifier(
                expressions=[expr],
                output_property='IsWaterOxygen'
            )
        )

        return pipeline

    def get_water_oxygen_indices(self, num_frame: int) -> np.ndarray:
        """Return the indices of oxygen atoms belonging to water molecules."""
        data = self.pipeline.compute(num_frame)
        if "IsWaterOxygen" in data.particles:
            mask = np.array(data.particles["IsWaterOxygen"].array) == 1
            return np.where(mask)[0]
        else:
            return self._manual_identification(data)

    def get_water_oxygen_positions(self, num_frame: int) -> np.ndarray:
        """Return XYZ coordinates of the oxygen atoms in water molecules."""
        indices = self.get_water_oxygen_indices(num_frame)
        data = self.pipeline.compute(num_frame)
        positions = np.asarray(data.particles["Position"])
        return positions[indices]

    def _manual_identification(self, data) -> np.ndarray:
        """Fallback to manual detection if OVITO fails."""
        positions = np.asarray(data.particles["Position"])
        types = np.asarray(data.particles["Particle Type"])
        oxygen_indices = np.where(types == self.oxygen_type)[0]
        hydrogen_indices = np.where(types == self.hydrogen_type)[0]
        water_oxygens = []

        for o_idx in oxygen_indices:
            o_pos = positions[o_idx]
            h_count = 0
            for h_idx in hydrogen_indices:
                if np.linalg.norm(o_pos - positions[h_idx]) <= self.oh_cutoff:
                    h_count += 1
            if h_count == 2:
                water_oxygens.append(o_idx)

        return np.array(water_oxygens)


# Example usage:
# Example usage of WaterMoleculeFinder and DumpParser

# # Step 1: Initialize WaterMoleculeFinder to identify water oxygen atoms
# water_finder = WaterMoleculeFinder(
#     in_path='traj_10_3_330w_nve_4k_reajust.lammpstrj',
#     particle_type_wall={3},
#     oxygen_type=1,
#     hydrogen_type=2
# )

# # Step 2: Get indices of water oxygen atoms for a specific frame
# num_frame = 0
# oxygen_indices = water_finder.get_water_oxygen_indices(num_frame=num_frame)
# print(f"Indices of water oxygen atoms in frame {num_frame}: {oxygen_indices}")

# # Step 3: Initialize DumpParser to parse the trajectory file
# parser = DumpParser(
#     in_path='traj_10_3_330w_nve_4k_reajust.lammpstrj',
#     particle_type_wall={3}
# )

# # Step 4: Parse the positions of water oxygen atoms for a specific frame
# frame_to_parse = 2
# oxygen_positions = parser.parse(num_frame=frame_to_parse, indices=oxygen_indices)
# print(f"Positions of water oxygen atoms in frame {frame_to_parse}:\n{oxygen_positions}")

# # Step 5: Print the number of water oxygen atoms
# num_water_oxygens = len(oxygen_positions)
# print(f"Number of water oxygen atoms in frame {frame_to_parse}: {num_water_oxygens}")

# parser = DumpParser('path/to/dumpfile.dump')
# positions = parser.parse(num_frame=10
# box_dimensions = parser.box_size(num_frame=10)