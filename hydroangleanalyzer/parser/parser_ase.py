from ase.io.trajectory import Trajectory
from ase.io import read, write
import numpy as np
from ase.neighborlist import NeighborList, natural_cutoffs
from .base_parser import BaseParser
from typing import Union, List, Set, Optional, Any
class Ase_Parser(BaseParser):
    def __init__(
        self,
        in_path: str,
        particle_type_wall: Union[List[str], Set[str]]
    ) -> None:
        """
        Initialize the Ase_Parser.

        Args:
            in_path (str): Path to input trajectory (any ASE-readable format)
            particle_type_wall (Union[List[str], Set[str]]): List or set of particle types (symbols) for wall particles
        """
        self.in_path: str = in_path
        self.particle_type_wall: Union[List[str], Set[str]] = particle_type_wall
        self.trajectory = read(self.in_path, index=':')

    def parse(self, num_frame, indices=None):
        """Return positions of particles for a specific frame, based on atoms indices"""
        frame = self.trajectory[num_frame]
        if indices is not None:
            # Ensure indices is a numpy array for consistent handling
            indices = np.array(indices)
            # Extract positions of particles based on the provided indices
            X_par = frame.positions[indices]
        else:
            # If no indices are provided, return all particle positions
            X_par = frame.positions

        return X_par

    def parse_liquid(self, particle_type_liquid, num_frame):
        """Return positions of liquid particles for a specific frame, excluding wall particles."""
        frame = self.trajectory[num_frame]
        # Create a boolean mask to include only liquid particles
        mask = np.isin(frame.symbols, particle_type_liquid)

        # Extract positions of liquid particles
        X_par = frame.positions[mask]
        return X_par
        
    def return_cylindrical_coord_pars(self, frame_list, type_model="masspain_y", liquid_indices=None):
        """Convert Cartesian coordinates to cylindrical coordinates for the given frames and indices."""
        xi_par = np.array([])
        zi_par = np.array([])

        for frame_idx in frame_list:
            frame = self.trajectory[frame_idx]
            X_par = frame.positions

            # Filter particles based on liquid_indices if provided
            if liquid_indices is not None:
                liquid_indices = np.array(liquid_indices)
                X_par = X_par[liquid_indices]

            X_cm = np.mean(X_par, axis=0)
            X_0 = X_par - X_cm
            X_0[:, 2] = X_par[:, 2]  # Keep z-coordinate unchanged

            if type_model == "masspain_y":
                xi_par_frame = np.abs(X_0[:, 0] + 0.01)
            elif type_model == "masspain_x":
                xi_par_frame = np.abs(X_0[:, 1] + 0.01)
            elif type_model == "spherical":
                xi_par_frame = np.sqrt(X_0[:, 0]**2 + X_0[:, 1]**2)

            zi_par_frame = X_0[:, 2]

            xi_par = np.concatenate((xi_par, xi_par_frame))
            zi_par = np.concatenate((zi_par, zi_par_frame))

            if frame_idx % 10 == 0:
                print(f"Frame: {frame_idx}")
                print(f"Center of Mass: {X_cm}")

        print("\nxi range:\t({},{})".format(np.min(xi_par), np.max(xi_par)))
        print("zi range:\t({},{})".format(np.min(zi_par), np.max(zi_par)))

        return xi_par, zi_par, len(frame_list)
   
    def box_size_y(self, num_frame):
        """Return the y-dimension of the simulation box for a specific frame."""
        frame = self.trajectory[num_frame]
        return frame.cell[1, 1]

    def box_size_x(self, num_frame):
        """Return the x-dimension of the simulation box for a specific frame."""
        frame = self.trajectory[num_frame]
        print(frame.cell)
        return frame.cell[0, 0]

    def box_lenght_max(self, num_frame):
        """Return the maximum dimension of the simulation box for a specific frame."""
        frame = self.trajectory[num_frame]
        return max(frame.cell.lengths())

    def frame_tot(self):
        """Return the total number of frames in the trajectory."""
        return len(self.trajectory)

class ASE_WaterMoleculeFinder:
    """Identify water oxygen atoms in an ASE trajectory."""

    def __init__(
        self,
        in_path: str,
        particle_type_wall: List[str],
        oxygen_type: str = 'O',
        hydrogen_type: str = 'H',
        oh_cutoff: float = 1.2,
    ):
        """
        Initialize the water molecule finder.

        Args:
            in_path: Path to the input trajectory file.
            particle_type_wall: List of symbols for wall particles.
            oxygen_type: Symbol for oxygen (default is 'O').
            hydrogen_type: Symbol for hydrogen (default is 'H').
            oh_cutoff: O–H distance cutoff for identifying water molecules.
        """
        self.trajectory = read(in_path, index=":")
        self.particle_type_wall = particle_type_wall
        self.oxygen_type = oxygen_type
        self.hydrogen_type = hydrogen_type
        self.oh_cutoff = oh_cutoff
    def get_water_oxygen_indices(self, num_frame):
        """Get indices of water oxygen atoms (based on having exactly two H neighbors)."""
        frame = self.trajectory[num_frame]
        atoms = frame
        symbols = np.array(frame.get_chemical_symbols())

        oxygen_indices = np.where(symbols == self.oxygen_type)[0]
        hydrogen_indices = np.where(symbols == self.hydrogen_type)[0]

        cutoffs = [self.oh_cutoff] * len(atoms)
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
        nl.update(atoms)

        water_oxygens = []
        for o_idx in oxygen_indices:
            indices, offsets = nl.get_neighbors(o_idx)
            h_neighbors = [i for i in indices if i in hydrogen_indices]
            if len(h_neighbors) == 2:
                water_oxygens.append(o_idx)

        return np.array(water_oxygens, dtype=int)

    def get_water_oxygen_positions(self, num_frame: int) -> np.ndarray:
        """
        Return XYZ coordinates of the oxygen atoms in water molecules for a specific frame.

        Args:
            frame_index: Index of the frame in the trajectory.

        Returns:
            np.ndarray: Positions of water oxygen atoms.
        """
        indices = self.get_water_oxygen_indices(num_frame)
        atoms = self.trajectory[num_frame]
        return atoms.positions[indices]


    def get_water_oxygen_positions(self, num_frame: int) -> np.ndarray:
        """
        Return XYZ coordinates of the oxygen atoms in water molecules for a specific frame.

        Args:
            frame_index: Index of the frame in the trajectory.

        Returns:
            np.ndarray: Positions of water oxygen atoms.
        """
        indices = self.get_water_oxygen_indices(num_frame)
        atoms = self.trajectory[num_frame]
        return atoms.positions[indices]

class Ase_wallParser:

    def __init__(self, in_path, particule_liquid_type):
        self.in_path = in_path
        self.particule_liquid_type = particule_liquid_type  # List of particle types to exclude (liquid particles)
        self.trajectory = read(self.in_path, index=':')

    def parse(self, num_frame):
        """Parse frame and return positions of wall particles."""
        frame = self.trajectory[num_frame]

        # Filter out liquid particles
        mask = ~np.isin(frame.get_chemical_symbols(), self.particule_liquid_type)
        X_par = frame.positions[mask]

        return X_par

    def find_highest_wall_part(self, num_frame):
        """Find the highest z-coordinate of wall particles."""
        X_wall = self.parse(num_frame)
        return np.max(X_wall[:, 2])

    def return_cylindrical_coord_pars(self, frame_list, type_model="masspain"):
        """Convert Cartesian coordinates to cylindrical coordinates for the given frames."""
        xi_par = np.array([])
        zi_par = np.array([])

        for frame in frame_list:
            X_par = self.parse(frame)
            dim = X_par.shape[1]
            X_cm = np.mean(X_par, axis=0)

            X_0 = X_par - X_cm
            X_0[:, 2] = X_par[:, 2]  # Keep z-coordinate unchanged

            if type_model == "masspain":
                xi_par_frame = np.abs(X_0[:, 0] + 0.01)
            elif type_model == "spherical":
                xi_par_frame = np.sqrt(X_0[:, 0]**2 + X_0[:, 1]**2)

            zi_par_frame = X_0[:, 2]

            xi_par = np.concatenate((xi_par, xi_par_frame))
            zi_par = np.concatenate((zi_par, zi_par_frame))

            if frame % 10 == 0:
                print(f"frame: {frame}")
                print(f"Center of Mass: {X_cm}")

        print("\nxi range:\t({},{})".format(np.min(xi_par), np.max(xi_par)))
        print("zi range:\t({},{})".format(np.min(zi_par), np.max(zi_par)))

        return xi_par, zi_par, len(frame_list)

    def box_size_y(self, num_frame):
        """Return the y-dimension of the simulation box for a specific frame."""
        frame = self.trajectory[num_frame]
        return frame.cell[1, 1]

    def box_lenght_max(self, num_frame):
        """Return the maximum dimension of the simulation box for a specific frame."""
        frame = self.trajectory[num_frame]
        return max(frame.cell.lengths())

    def frame_tot(self):
        """Return the total number of frames in the trajectory."""
        return len(self.trajectory)
# Example usage:
# parser = DumpParser('path/to/dumpfile.dump')
# water_finder = Dump_WaterMoleculeFinder('path/to/dumpfile.dump', particle_type_wall=['Si', 'O'], oxygen_type='O', hydrogen_type='H')
# water_oxygen_indices = water_finder.get_water_oxygen_indices(num_frame=10)
# positions = parser.parse(num_frame=10
# box_dimensions = parser.box_size(num_frame=10)