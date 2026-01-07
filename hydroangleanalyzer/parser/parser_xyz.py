import numpy as np

from .base_parser import BaseParser


class XYZParser(BaseParser):
    """Simple in-memory XYZ trajectory parser with lattice extraction.

    Parameters
    ----------
    in_path : str
        Path to extended XYZ trajectory containing per-frame atom count, comment line
        with lattice vectors, then atom symbol + coordinates.
    """

    def __init__(self, in_path):
        self.in_path = in_path
        self.frames = self.load_xyz_file()

    def load_xyz_file(self):
        """Load all frames from the XYZ file into memory.

        Returns
        -------
        list[dict]
            Each entry has keys: ``symbols``, ``positions``, ``lattice_matrix``.
        """
        frames = []
        with open(self.in_path, "r") as file:
            lines = file.readlines()
        frame_start = 0
        while frame_start < len(lines):
            num_atoms = int(lines[frame_start].strip())
            frame_start += 1
            comment_line = lines[frame_start].strip()
            lattice_info = comment_line.split('Lattice="')[1].split('"')[0]
            lattice_vectors = np.array(lattice_info.split(), dtype=float)
            lattice_matrix = lattice_vectors.reshape(3, 3)
            frame_start += 1
            symbols = []
            positions = []
            for i in range(num_atoms):
                parts = lines[frame_start + i].strip().split()
                symbols.append(parts[0])
                positions.append([float(coord) for coord in parts[1:4]])
            frames.append(
                {
                    "symbols": np.array(symbols),
                    "positions": np.array(positions),
                    "lattice_matrix": lattice_matrix,
                }
            )
            frame_start += num_atoms
        return frames

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
            Coordinates of requested atoms.
        """
        frame = self.frames[num_frame]
        if indices is not None:
            indices = np.array(indices)
            return frame["positions"][indices]
        return frame["positions"]

    def parse_liquid(self, particle_type_liquid, num_frame):
        """Return positions of liquid particles (filter by symbols).

        Parameters
        ----------
        particle_type_liquid : sequence[str]
            Atom symbols considered liquid (e.g. water molecule constituents).
        num_frame : int
            Frame index.

        Returns
        -------
        ndarray, shape (L, 3)
            Cartesian coordinates of liquid atoms.
        """
        frame = self.frames[num_frame]
        mask = np.isin(frame["symbols"], particle_type_liquid)
        return frame["positions"][mask]

    def return_cylindrical_coord_pars(
        self, frame_list, type_model="cylinder_x", liquid_indices=None
    ):
        """Convert selected frames to cylindrical coordinates per axis mode.

        Parameters
        ----------
        frame_list : sequence[int]
            Frame indices to process.
        type_model : str, default "cylinder_x"
            One of {"cylinder_y", "cylinder_x", "spherical"} controlling
            radial definition.
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
            frame = self.frames[frame_idx]
            X_par = frame["positions"]
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

    def box_lenght_max(self, num_frame):  # legacy spelling retained
        """Return maximum lattice vector length (legacy name)."""
        return self.box_length_max(num_frame)

    def box_length_max(self, num_frame):
        """Return the maximum lattice vector length for a frame.

        Parameters
        ----------
        num_frame : int
            Frame index.

        Returns
        -------
        float
            Max |a_i| over lattice vectors.
        """
        lattice_matrix = self.frames[num_frame]["lattice_matrix"]
        return float(np.max(np.linalg.norm(lattice_matrix, axis=1)))

    def box_size_x(self, num_frame):
        """Return box length along x (a1x component)."""
        lattice_matrix = self.frames[num_frame]["lattice_matrix"]
        return float(lattice_matrix[0, 0])

    def box_size_y(self, num_frame):
        """Return box length along y (a2y component)."""
        lattice_matrix = self.frames[num_frame]["lattice_matrix"]
        return float(lattice_matrix[1, 1])

    def frame_tot(self):
        """Return total number of frames loaded."""
        return len(self.frames)


class XYZWaterMoleculeFinder(BaseParser):
    """Parser specialized for identifying water oxygen atoms in XYZ trajectories.

    Parameters
    ----------
    in_path : str
        Path to XYZ file.
    particle_type_wall : sequence[str]
        Symbols that represent wall (excluded) particles.
    oxygen_type : str, default "O"
        Oxygen atom symbol.
    hydrogen_type : str, default "H"
        Hydrogen atom symbol.
    oh_cutoff : float, default 1.2
        Distance cutoff (Å) for O-H bonding to identify water molecules.
    """

    def __init__(
        self,
        in_path,
        particle_type_wall,
        oxygen_type="O",
        hydrogen_type="H",
        oh_cutoff=1.2,
    ):
        self.in_path = in_path
        self.particle_type_wall = particle_type_wall
        self.oxygen_type = oxygen_type
        self.hydrogen_type = hydrogen_type
        self.oh_cutoff = oh_cutoff
        self.frames = self.load_xyz_file()

    def load_xyz_file(self):
        """Load frames (without lattice) for water oxygen analysis."""
        frames = []
        with open(self.in_path, "r") as file:
            lines = file.readlines()
        frame_start = 0
        while frame_start < len(lines):
            num_atoms = int(lines[frame_start].strip())
            frame_start += 1
            frame_start += 1  # skip comment
            symbols = []
            positions = []
            for i in range(num_atoms):
                parts = lines[frame_start + i].strip().split()
                symbols.append(parts[0])
                positions.append([float(coord) for coord in parts[1:4]])
            frames.append(
                {"symbols": np.array(symbols), "positions": np.array(positions)}
            )
            frame_start += num_atoms
        return frames

    def parse(self, particle_type_liquid, num_frame):
        """Return liquid particle coordinates filtering wall types.

        Parameters
        ----------
        particle_type_liquid : sequence[str]
            Symbols for liquid particles.
        num_frame : int
            Frame index.

        Returns
        -------
        ndarray, shape (L, 3)
            Liquid atom positions.
        """
        frame = self.frames[num_frame]
        mask = np.isin(frame["symbols"], particle_type_liquid)
        return frame["positions"][mask]

    def return_cylindrical_coord_pars(
        self, frame_list, type_model="cylinder_y", liquid_indices=None
    ):
        """Return cylindrical coordinate arrays for multiple frames.

        Parameters
        ----------
        frame_list : sequence[int]
            Frames to process.
        type_model : str, default "cylinder_y"
            One of {"cylinder_y", "cylinder_x", "spherical"}.
        liquid_indices : sequence[int], optional
            Subset indices for atoms of interest.

        Returns
        -------
        tuple(ndarray, ndarray, int)
            (xi_values, zi_values, n_frames).
        """
        xi_par = np.array([])
        zi_par = np.array([])
        for frame_idx in frame_list:
            frame = self.frames[frame_idx]
            X_par = frame["positions"]
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

    def box_lenght_max(self, num_frame):  # legacy name retained
        """XYZ lacks lattice; method unsupported (legacy name)."""
        raise NotImplementedError("XYZ files do not inherently store box dimensions.")

    def get_water_oxygen_indices(self, num_frame):
        """Return indices of oxygen atoms belonging to water molecules.

        Parameters
        ----------
        num_frame : int
            Frame index.

        Returns
        -------
        ndarray
            Indices of oxygen atoms with exactly two hydrogens within cutoff.
        """
        data = self.frames[num_frame]
        positions = data["positions"]
        symbols = data["symbols"]
        oxygen_indices = np.where(symbols == self.oxygen_type)[0]
        hydrogen_indices = np.where(symbols == self.hydrogen_type)[0]
        return self._manual_water_identification(
            positions, oxygen_indices, hydrogen_indices
        )

    def get_water_oxygen_positions(self, num_frame):
        """Return coordinates of water oxygen atoms for a frame.

        Parameters
        ----------
        num_frame : int
            Frame index.

        Returns
        -------
        ndarray, shape (N, 3)
            Oxygen atom positions; empty array if none detected.
        """
        positions = self.frames[num_frame]["positions"]
        indices = self.get_water_oxygen_indices(num_frame)
        if len(indices) == 0:
            return np.empty((0, 3))
        return positions[indices]

    def _manual_water_identification(self, positions, oxygen_indices, hydrogen_indices):
        """Identify water oxygens by counting hydrogens within cutoff distance.

        Parameters
        ----------
        positions : ndarray, shape (N, 3)
            Atomic positions.
        oxygen_indices : ndarray
            Candidate oxygen indices.
        hydrogen_indices : ndarray
            Hydrogen indices to check.

        Returns
        -------
        ndarray
            Oxygen indices with exactly two nearby hydrogens.
        """
        water_oxygens = []
        for o_idx in oxygen_indices:
            o_pos = positions[o_idx]
            h_count = 0
            for h_idx in hydrogen_indices:
                h_pos = positions[h_idx]
                if np.linalg.norm(o_pos - h_pos) <= self.oh_cutoff:
                    h_count += 1
            if h_count == 2:
                water_oxygens.append(o_idx)
        return np.array(water_oxygens)


class XYZWallParser:
    """Parser for extracting wall particle coordinates from XYZ trajectories.

    Parameters
    ----------
    in_path : str
        Path to XYZ file.
    particule_liquid_type : sequence[str]
        Symbols representing liquid particles to exclude.
    """

    def __init__(self, in_path, particule_liquid_type):
        self.in_path = in_path
        self.particule_liquid_type = particule_liquid_type
        self.frames = self.load_xyz_file()

    def load_xyz_file(self):
        """Load frames (without lattice) for wall extraction."""
        frames = []
        with open(self.in_path, "r") as file:
            lines = file.readlines()
        frame_start = 0
        while frame_start < len(lines):
            num_atoms = int(lines[frame_start].strip())
            frame_start += 1
            frame_start += 1  # skip comment line
            symbols = []
            positions = []
            for i in range(num_atoms):
                parts = lines[frame_start + i].strip().split()
                symbols.append(parts[0])
                positions.append([float(coord) for coord in parts[1:4]])
            frames.append(
                {"symbols": np.array(symbols), "positions": np.array(positions)}
            )
            frame_start += num_atoms
        return frames

    def parse(self, num_frame):
        """Return coordinates of wall particles for frame (excludes liquid symbols)."""
        frame = self.frames[num_frame]
        mask = ~np.isin(frame["symbols"], self.particule_liquid_type)
        return frame["positions"][mask]

    def find_highest_wall_part(self, num_frame):
        """Return maximum z among wall particle positions for frame."""
        X_wall = self.parse(num_frame)
        return float(np.max(X_wall[:, 2]))

    def return_cylindrical_coord_pars(self, frame_list, type_model="cylinder"):
        """Return cylindrical projection arrays for wall particles across frames.

        Parameters
        ----------
        frame_list : sequence[int]
            Frame indices.
        type_model : str, default "cylinder"
            Either "cylinder" or "spherical" to select radial metric.

        Returns
        -------
        tuple(ndarray, ndarray, int)
            (xi_values, zi_values, n_frames).
        """
        xi_par = np.array([])
        zi_par = np.array([])
        for frame in frame_list:
            X_par = self.parse(frame)
            X_cm = np.mean(X_par, axis=0)
            X_0 = X_par - X_cm
            X_0[:, 2] = X_par[:, 2]
            if type_model == "cylinder":
                xi_frame = np.abs(X_0[:, 0] + 0.01)
            else:
                xi_frame = np.sqrt(X_0[:, 0] ** 2 + X_0[:, 1] ** 2)
            zi_frame = X_0[:, 2]
            xi_par = np.concatenate((xi_par, xi_frame))
            zi_par = np.concatenate((zi_par, zi_frame))
            if frame % 10 == 0:
                print(f"frame: {frame}\nCenter of Mass: {X_cm}")
        print(f"\nxi range:\t({np.min(xi_par)},{np.max(xi_par)})")
        print(f"zi range:\t({np.min(zi_par)},{np.max(zi_par)})")
        return xi_par, zi_par, len(frame_list)

    def box_size_y(self, num_frame):  # placeholders retained for interface parity
        """Placeholder: XYZ lacks intrinsic box dimension metadata (y)."""
        raise NotImplementedError("XYZ files do not inherently store box dimensions.")

    def box_lenght_max(self, num_frame):  # legacy spelling retained
        """Placeholder: XYZ lacks intrinsic box dimension metadata (max length)."""
        raise NotImplementedError("XYZ files do not inherently store box dimensions.")

    def frame_tot(self):
        """Return total number of frames loaded."""
        return len(self.frames)


XYZ_Parser = XYZParser
XYZ_WaterMoleculeFinder = XYZWaterMoleculeFinder
XYZ_WallParser = XYZWallParser
