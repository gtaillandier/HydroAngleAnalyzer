from ovito.io import import_file, export_file
from ovito.modifiers import (SelectTypeModifier, DeleteSelectedModifier, ComputePropertyModifier)

import numpy as np
from ovito.io import import_file, export_file
from ovito.modifiers import (SelectTypeModifier, DeleteSelectedModifier, ComputePropertyModifier,CoordinationAnalysisModifier )

class DumpParser:
    def __init__(self, in_path, particle_type_wall):
        self.in_path = in_path
        self.particle_type_wall = particle_type_wall
        self.pipeline = self.load_dump_ovito()

    def load_dump_ovito(self):
        # Load the file using OVITO
        pipeline = import_file(self.in_path)
        pipeline.modifiers.append(SelectTypeModifier(property='Particle Type', types=self.particle_type_wall))
        pipeline.modifiers.append(DeleteSelectedModifier())
        pipeline.modifiers.append(ComputePropertyModifier(expressions=['1'], output_property='Unity'))
        return pipeline

    def parse(self, num_frame):
        data = self.pipeline.compute(num_frame)
        X_par = np.asarray(data.particles["Position"])
        return X_par

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
class WaterOxygenDumpParser:
    def __init__(self, in_path, particle_type_wall, oxygen_type=3, hydrogen_type=2, oh_cutoff=1.2):
        self.in_path = in_path
        self.particle_type_wall = particle_type_wall
        self.oxygen_type = oxygen_type
        self.hydrogen_type = hydrogen_type
        self.oh_cutoff = oh_cutoff
        self.pipeline = self.load_dump_ovito()

    def load_dump_ovito(self):
        # Load the file using OVITO
        pipeline = import_file(self.in_path)

        # First remove wall particles
        pipeline.modifiers.append(SelectTypeModifier(
            property='Particle Type',
            types=self.particle_type_wall
        ))
        pipeline.modifiers.append(DeleteSelectedModifier())

        # Then add coordination analysis modifier
        coord_modifier = CoordinationAnalysisModifier(
            cutoff=self.oh_cutoff,
            number_of_bins=200
        )
        pipeline.modifiers.append(coord_modifier)

        # Add a compute property modifier to identify water oxygens
        water_oxygen_expr = f"""
        (ParticleType == {self.oxygen_type}) &&
        (Coordination == 2)
        """

        pipeline.modifiers.append(ComputePropertyModifier(
            expressions=[water_oxygen_expr],
            output_property='IsWaterOxygen'
        ))

        return pipeline

    def parse(self, num_frame):
        """Parse frame and return positions with water oxygen identification"""
        data = self.pipeline.compute(num_frame)

        positions = np.asarray(data.particles["Position"])
        particle_types = np.asarray(data.particles["Particle Type"])

        # Get water oxygen mask if the property exists
        water_oxygen_mask = None
        if "IsWaterOxygen" in data.particles:
            water_oxygen_mask = np.array(data.particles["IsWaterOxygen"].array) == 1
            
        return {
            'positions': positions,
            'types': particle_types,
            'water_oxygen_mask': water_oxygen_mask
        }

    def get_water_oxygen_indices(self, num_frame):
        """Get indices of water oxygen atoms"""
        data = self.parse(num_frame)
        if data['water_oxygen_mask'] is not None:
            return np.where(data['water_oxygen_mask'])[0]
        else:
            # Fallback to manual identification
            return self._manual_water_identification(data)
    def get_water_oxygen_positions(self, num_frame):
        """
        Returns the array of XYZ coordinates of oxygen atoms in water molecules.

        Args:
            num_frame (int): The frame number to analyze

        Returns:
            numpy.ndarray: Array of shape (N, 3) containing XYZ coordinates of water oxygen atoms
        """
        # Get the indices of water oxygen atoms
        water_oxygen_indices = self.get_water_oxygen_indices(num_frame)

        if len(water_oxygen_indices) == 0:
            return np.empty((0, 3))  # Return empty array if no water oxygens found

        # Get all positions and select only the water oxygen positions
        data = self.parse(num_frame)
        all_positions = data['positions']

        return all_positions[water_oxygen_indices]
        
    def _manual_water_identification(self, data):
        """Manual identification if OVITO modifiers don't work"""
        positions = data['positions']
        types = data['types']

        oxygen_indices = np.where(types == self.oxygen_type)[0]
        hydrogen_indices = np.where(types == self.hydrogen_type)[0]

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

class WaterOxygenDumpParser_old:
    def __init__(self, in_path, particle_type_wall, oxygen_type=1, hydrogen_type=3, oh_cutoff=1.2):
        self.in_path = in_path
        self.particle_type_wall = particle_type_wall
        self.oxygen_type = oxygen_type
        self.hydrogen_type = hydrogen_type
        self.oh_cutoff = oh_cutoff
        self.pipeline = self.load_dump_ovito()
        
    def load_dump_ovito(self):
        # Load the file using OVITO
        pipeline = import_file(self.in_path)
        
        # Add coordination analysis modifier to identify water oxygens
        coord_modifier = CoordinationAnalysisModifier(
            cutoff=self.oh_cutoff,
            number_of_bins=200
        )
        pipeline.modifiers.append(coord_modifier)
        
        # Add a compute property modifier to identify water oxygens
        # This will create a property that marks water oxygens
        water_oxygen_expr = f"""
        (ParticleType == {self.oxygen_type}) && 
        (Coordination == 2) && 
        (CoordinationNumber.{self.hydrogen_type} == 2)
        """
        
        pipeline.modifiers.append(ComputePropertyModifier(
            expressions=[water_oxygen_expr],
            output_property='IsWaterOxygen'
        ))
        
        # Remove wall particles
        pipeline.modifiers.append(SelectTypeModifier(
            property='Particle Type', 
            types=self.particle_type_wall
        ))
        pipeline.modifiers.append(DeleteSelectedModifier())
        
        return pipeline
    
    def parse(self, num_frame):
        """Parse frame and return positions with water oxygen identification"""
        data = self.pipeline.compute(num_frame)
        
        positions = np.asarray(data.particles["Position"])
        particle_types = np.asarray(data.particles["Particle Type"])
        
        # Get water oxygen mask if the property exists
        water_oxygen_mask = None
        if "IsWaterOxygen" in data.particles:
            water_oxygen_mask = np.asarray(data.particles["IsWaterOxygen"], dtype=bool)
        
        return {
            'positions': positions,
            'types': particle_types,
            'water_oxygen_mask': water_oxygen_mask
        }
    
    def get_water_oxygen_indices(self, num_frame):
        """Get indices of water oxygen atoms"""
        data = self.parse(num_frame)
        if data['water_oxygen_mask'] is not None:
            return np.where(data['water_oxygen_mask'])[0]
        else:
            # Fallback to manual identification
            return self._manual_water_identification(data)
    
    def _manual_water_identification(self, data):
        """Manual identification if OVITO modifiers don't work"""
        positions = data['positions']
        types = data['types']
        
        oxygen_indices = np.where(types == self.oxygen_type)[0]
        hydrogen_indices = np.where(types == self.hydrogen_type)[0]
        
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


# Example usage:
# parser = DumpParser('path/to/dumpfile.dump')
# positions = parser.parse(num_frame=10
# box_dimensions = parser.box_size(num_frame=10)