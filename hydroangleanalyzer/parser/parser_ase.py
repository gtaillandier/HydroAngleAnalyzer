#ase parser
from ase.io.trajectory import Trajectory
from ase.io import read, write
import numpy as np
from ase.neighborlist import NeighborList, natural_cutoffs

class Ase_Parser:
    def __init__(self, in_path, particle_type_wall):
        self.in_path = in_path
        self.particle_type_wall = particle_type_wall
        self.particle_liquid_type = particle_type_liquid
        self.trajectory = read(self.in_path, index=':')

    def parse(self, num_frame):
        """Return positions of particles for a specific frame, excluding oxygen atoms."""
        frame = self.trajectory[num_frame]
        # Filter out oxygen atoms (assuming 'O' is stored in the 'symbols' array)
        mask = frame.symbols != 'O'
        X_par = frame.positions[mask]
        return X_par
    
    def parse_liquid(self, num_frame):
        """Return positions of liquid particles for a specific frame, excluding wall particles."""
        frame = self.trajectory[num_frame]
        # Filter out wall particles
        mask = ~np.isin(frame.symbols, self.particle_type_wall)
        X_par = frame.positions[mask]
        return X_par

    def return_cylindrical_coord_pars(self, frame_list, type_model="masspain"):
        """Convert Cartesian coordinates to cylindrical coordinates for the given frames."""
        xi_par = np.array([])
        zi_par = np.array([])

        for frame_idx in frame_list:
            frame = self.trajectory[frame_idx]
            # Filter out oxygen atoms
            mask = frame.symbols != ['O']
            X_par = frame.positions[mask]

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

    def box_lenght_max(self, num_frame):
        """Return the maximum dimension of the simulation box for a specific frame."""
        frame = self.trajectory[num_frame]
        return max(frame.cell.lengths())

    def frame_tot(self):
        """Return the total number of frames in the trajectory."""
        return len(self.trajectory)


class Ase_WaterOxygenParser:
    def __init__(self, in_path, particle_type_wall, oxygen_type='O', hydrogen_type='H', oh_cutoff=1.2):
        """
        ASE-based parser that identifies water oxygen atoms using NeighborList.

        Args:
            in_path (str): Path to input trajectory (any ASE-readable format)
            particle_type_wall (list): List of particle types (symbols) to exclude (e.g. ['C', 'Si'])
            oxygen_type (str): Atomic symbol of oxygen (default 'O')
            hydrogen_type (str): Atomic symbol of hydrogen (default 'H')
            oh_cutoff (float): O–H cutoff distance (Å)
        """
        self.in_path = in_path
        self.particle_type_wall = particle_type_wall
        self.oxygen_type = oxygen_type
        self.hydrogen_type = hydrogen_type
        self.oh_cutoff = oh_cutoff
        self.trajectory = read(self.in_path, index=':')

    def parse(self, num_frame):
        """Parse frame and return positions and symbols (excluding wall particles)."""
        frame = self.trajectory[num_frame]

        # Exclude wall particles
        mask = ~np.isin(frame.get_chemical_symbols(), self.particle_type_wall)
        positions = frame.positions[mask]
        symbols = np.array(frame.get_chemical_symbols())[mask]

        # Build a reduced ASE atoms object with the filtered atoms
        from ase import Atoms
        frame_filtered = Atoms(symbols=symbols, positions=positions, cell=frame.cell, pbc=frame.pbc)

        return {
            'positions': positions,
            'symbols': symbols,
            'atoms': frame_filtered
        }

    def get_water_oxygen_indices(self, num_frame):
        """Get indices of water oxygen atoms (based on having exactly two H neighbors)."""
        data = self.parse(num_frame)
        atoms = data['atoms']
        symbols = data['symbols']

        # Identify oxygen and hydrogen indices
        oxygen_indices = np.where(symbols == self.oxygen_type)[0]
        hydrogen_indices = np.where(symbols == self.hydrogen_type)[0]
        # Build neighbor list using ASE
        cutoffs = [self.oh_cutoff] * len(atoms)
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
        nl.update(atoms)

        water_oxygens = []

        for o_idx in oxygen_indices:
            # Get neighbors of the oxygen atom
            indices, offsets = nl.get_neighbors(o_idx)
            # Filter only hydrogens among the neighbors
            h_neighbors = [i for i in indices if i in hydrogen_indices]
            # A water oxygen has exactly two H neighbors
            if len(h_neighbors) == 2:
                water_oxygens.append(o_idx)

        return np.array(water_oxygens, dtype=int)

    def get_water_oxygen_positions(self, num_frame):
        """
        Returns XYZ coordinates of oxygen atoms in water molecules.

        Args:
            num_frame (int): Frame number to analyze

        Returns:
            numpy.ndarray: Array of shape (N, 3) with water oxygen coordinates
        """
        data = self.parse(num_frame)
        positions = data['positions']
        water_oxygen_indices = self.get_water_oxygen_indices(num_frame)

        if len(water_oxygen_indices) == 0:
            return np.empty((0, 3))

        return positions[water_oxygen_indices]

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
# positions = parser.parse(num_frame=10
# box_dimensions = parser.box_size(num_frame=10)