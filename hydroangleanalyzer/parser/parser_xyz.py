import num as np
class XYZ_Parser:
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
        """Parse frame and return positions with particle symbols."""
        frame = self.frames[num_frame]

        # Filter out wall particles
        mask = ~np.isin(frame['symbols'], self.particle_type_wall)
        positions = frame['positions'][mask]
        symbols = frame['symbols'][mask]

        return {
            'positions': positions,
            'symbols': symbols,
        }

    def get_water_oxygen_indices(self, num_frame, oxygen_type='O', hydrogen_type='H', oh_cutoff=1.2):
        """Get indices of water oxygen atoms."""
        data = self.parse(num_frame)
        positions = data['positions']
        symbols = data['symbols']

        # Get oxygen and hydrogen indices
        oxygen_indices = np.where(symbols == oxygen_type)[0]
        hydrogen_indices = np.where(symbols == hydrogen_type)[0]

        water_oxygens = self._manual_water_identification(positions, oxygen_indices, hydrogen_indices, oh_cutoff)
        return water_oxygens

    def get_water_oxygen_positions(self, num_frame, oxygen_type='O', hydrogen_type='H', oh_cutoff=1.2):
        """
        Returns the array of XYZ coordinates of oxygen atoms in water molecules.
        Args:
            num_frame (int): The frame number to analyze
        Returns:
            numpy.ndarray: Array of shape (N, 3) containing XYZ coordinates of water oxygen atoms
        """
        data = self.parse(num_frame)
        positions = data['positions']

        water_oxygen_indices = self.get_water_oxygen_indices(num_frame, oxygen_type, hydrogen_type, oh_cutoff)
        if len(water_oxygen_indices) == 0:
            return np.empty((0, 3))  # Return empty array if no water oxygens found

        return positions[water_oxygen_indices]

    def _manual_water_identification(self, positions, oxygen_indices, hydrogen_indices, oh_cutoff):
        """Manual identification of water oxygen atoms."""
        water_oxygens = []

        for o_idx in oxygen_indices:
            o_pos = positions[o_idx]
            h_count = 0

            # Count hydrogens within cutoff distance
            for h_idx in hydrogen_indices:
                h_pos = positions[h_idx]
                distance = np.linalg.norm(o_pos - h_pos)
                if distance <= oh_cutoff:
                    h_count += 1

            # Water oxygens should have exactly 2 hydrogens
            if h_count == 2:
                water_oxygens.append(o_idx)

        return np.array(water_oxygens)