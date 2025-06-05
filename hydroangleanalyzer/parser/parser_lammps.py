#parser lammps traj
#ase parser
from ovito.io import import_file, export_file
from ovito.modifiers import (SelectTypeModifier, DeleteSelectedModifier, ComputePropertyModifier)

import numpy as np
from ovito.io import import_file, export_file
from ovito.modifiers import (SelectTypeModifier, DeleteSelectedModifier, ComputePropertyModifier)

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
    


# Example usage:
# parser = DumpParser('path/to/dumpfile.dump')
# positions = parser.parse(num_frame=10
# box_dimensions = parser.box_size(num_frame=10)