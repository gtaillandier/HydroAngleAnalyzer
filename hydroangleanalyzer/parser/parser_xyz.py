import numpy as np
from .base_parser import BaseParser

class XYZ_Parser(BaseParser):
    def __init__(self, in_path, particle_type_wall):
        self.in_path = in_path
        self.particle_type_wall = particle_type_wall  # List of particle types to exclude
        self.frames = self.load_xyz_file()

    def load_xyz_file(self):

        """Load and parse the XYZ file."""

        frames = []

        with open(self.in_path, 'r') as file:

            lines = file.readlines()

        frame_start = 0

        while frame_start < len(lines):

            # Read the number of atoms

            num_atoms = int(lines[frame_start].strip())

            frame_start += 1

            # Parse the comment line for lattice information

            comment_line = lines[frame_start].strip()

            lattice_info = comment_line.split('Lattice="')[1].split('"')[0]

            lattice_vectors = np.array(lattice_info.split(), dtype=float)

            lattice_matrix = lattice_vectors.reshape(3, 3)

            frame_start += 1

            # Read atom data

            symbols = []

            positions = []

            for i in range(num_atoms):

                parts = lines[frame_start + i].strip().split()

                symbols.append(parts[0])

                positions.append([float(coord) for coord in parts[1:4]])

            frames.append({

                'symbols': np.array(symbols),

                'positions': np.array(positions),

                'lattice_matrix': lattice_matrix

            })

            frame_start += num_atoms

        return frames

    def parse(self, num_frame, indices=None):
        """
        Return positions of particles for a specific frame, based on atom indices.

        Args:
            num_frame (int): The index of the frame to parse.
            indices (list or numpy.ndarray, optional): List of indices to extract. If None, all particle positions are returned.

        Returns:
            numpy.ndarray: Coordinates of the specified particles.
        """
        frame = self.frames[num_frame]
        if indices is not None:
            indices = np.array(indices)
            X_par = frame['positions'][indices]
        else:
            X_par = frame['positions']
        return X_par
    def box_size_x(self, num_frame):
        """
        Return the box size in the x direction for a specific frame.
        Args:
            num_frame (int): The index of the frame.
        Returns:
            float: The box size in the x direction.
        """
        lattice_matrix = self.frames[num_frame]['lattice_matrix']
        return lattice_matrix[0, 0]

    def box_size_y(self, num_frame):
        """
        Return the box size in the y direction for a specific frame.
        Args:
            num_frame (int): The index of the frame.
        Returns:
            float: The box size in the y direction.
        """
        lattice_matrix = self.frames[num_frame]['lattice_matrix']
        return lattice_matrix[1, 1]

    def frame_tot(self):
        """Return the total number of frames in the trajectory."""
        return len(self.frames)
        

class XYZ_WaterOxygenParser(BaseParser):
    def __init__(self, in_path, particle_type_wall, oxygen_type='O', hydrogen_type='H', oh_cutoff=1.2):
        self.in_path = in_path
        self.particle_type_wall = particle_type_wall  # List of particle types to exclude
        self.oxygen_type = oxygen_type
        self.hydrogen_type = hydrogen_type
        self.oh_cutoff = oh_cutoff
        self.frames = self.load_xyz_file()

    def load_xyz_file(self):
        """Load and parse the XYZ file."""
        frames = []
        with open(self.in_path, 'r') as file:
            lines = file.readlines()

        frame_start = 0
        while frame_start < len(lines):
            # Read the number of atoms
            num_atoms = int(lines[frame_start].strip())
            frame_start += 1

            # Skip the comment line
            frame_start += 1

            # Read atom data
            symbols = []
            positions = []
            for i in range(num_atoms):
                parts = lines[frame_start + i].strip().split()
                symbols.append(parts[0])
                positions.append([float(coord) for coord in parts[1:4]])

            frames.append({
                'symbols': np.array(symbols),
                'positions': np.array(positions)
            })

            frame_start += num_atoms

        return frames

    def parse(self, num_frame):
        """Parse frame and return positions with water oxygen identification."""
        frame = self.frames[num_frame]

        # Filter out wall particles
        mask = ~np.isin(frame['symbols'], self.particle_type_wall)
        positions = frame['positions'][mask]
        symbols = frame['symbols'][mask]

        return {
            'positions': positions,
            'symbols': symbols,
        }

    def get_water_oxygen_indices(self, num_frame):
        """Get indices of water oxygen atoms."""
        data = self.parse(num_frame)
        positions = data['positions']
        symbols = data['symbols']

        # Get oxygen and hydrogen indices
        oxygen_indices = np.where(symbols == self.oxygen_type)[0]
        hydrogen_indices = np.where(symbols == self.hydrogen_type)[0]

        water_oxygens = self._manual_water_identification(positions, oxygen_indices, hydrogen_indices)
        return water_oxygens

    def get_water_oxygen_positions(self, num_frame):
        """
        Returns the array of XYZ coordinates of oxygen atoms in water molecules.
        Args:
            num_frame (int): The frame number to analyze
        Returns:
            numpy.ndarray: Array of shape (N, 3) containing XYZ coordinates of water oxygen atoms
        """
        data = self.parse(num_frame)
        positions = data['positions']

        water_oxygen_indices = self.get_water_oxygen_indices(num_frame)
        if len(water_oxygen_indices) == 0:
            return np.empty((0, 3))  # Return empty array if no water oxygens found

        return positions[water_oxygen_indices]

    def _manual_water_identification(self, positions, oxygen_indices, hydrogen_indices):
        """Manual identification of water oxygen atoms."""
        water_oxygens = []

        for o_idx in oxygen_indices:
            o_pos = positions[o_idx]
            h_count = 0

            # Count hydrogens within cutoff distance
            for h_idx in hydrogen_indices:
                h_pos = positions[h_idx]
                distance = np.linalg.norm(o_pos - h_pos)
                if distance <= self.oh_cutoff:
                    h_count += 1

            # Water oxygens should have exactly 2 hydrogens
            if h_count == 2:
                water_oxygens.append(o_idx)

        return np.array(water_oxygens)
class XYZ_wallParser:
    def __init__(self, in_path, particule_liquid_type):
        self.in_path = in_path
        self.particule_liquid_type = particule_liquid_type  # List of particle types to exclude (liquid particles)
        self.frames = self.load_xyz_file()

    def load_xyz_file(self):
        """Load and parse the XYZ file."""
        frames = []
        with open(self.in_path, 'r') as file:
            lines = file.readlines()

        frame_start = 0
        while frame_start < len(lines):
            # Read the number of atoms
            num_atoms = int(lines[frame_start].strip())
            frame_start += 1

            # Skip the comment line
            frame_start += 1

            # Read atom data
            symbols = []
            positions = []
            for i in range(num_atoms):
                parts = lines[frame_start + i].strip().split()
                symbols.append(parts[0])
                positions.append([float(coord) for coord in parts[1:4]])

            frames.append({
                'symbols': np.array(symbols),
                'positions': np.array(positions)
            })

            frame_start += num_atoms

        return frames

    def parse(self, num_frame):
        """Parse frame and return positions of wall particles."""
        frame = self.frames[num_frame]

        # Filter out liquid particles
        mask = ~np.isin(frame['symbols'], self.particule_liquid_type)
        X_par = frame['positions'][mask]

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
        # XYZ files do not inherently store box dimensions, so this method is a placeholder.
        # You might need to modify this based on how box dimensions are stored in your XYZ files.
        raise NotImplementedError("XYZ files do not inherently store box dimensions.")

    def box_lenght_max(self, num_frame):
        """Return the maximum dimension of the simulation box for a specific frame."""
        # XYZ files do not inherently store box dimensions, so this method is a placeholder.
        # You might need to modify this based on how box dimensions are stored in your XYZ files.
        raise NotImplementedError("XYZ files do not inherently store box dimensions.")

    def frame_tot(self):
        """Return the total number of frames in the trajectory."""
        return len(self.frames)