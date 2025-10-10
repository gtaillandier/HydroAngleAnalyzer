import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from .surface_definition import HyperbolicTangentModel

class ContactAngleAnalyzer:
    """
    Generalized class for analyzing contact angles in MD simulations using binning.
    Compatible with any parser implementing BaseParser.
    """

    def __init__(
        self,
        parser,
        liquid_indices=None,
        wall_height=None,
        type_model="spherical",
        width_masspain=21,
        binning_params=None,
        output_dir="output_analysis/"
    ):
        """
        Parameters
        ----------
        parser : BaseParser
            Parser object for reading trajectory data.
        liquid_indices : array-like, optional
            Indices of liquid particles (if needed by parser).
        wall_height : float, optional
            Height of the wall surface (if applicable).
        type_model : str
            Type of model for volume calculation ("spherical" or "masspain").
        width_masspain : float
            Width parameter for masspain model.
        binning_params : dict, optional
            Dict with binning parameters.
        output_dir : str
            Directory for output files.
        """
        self.parser = parser
        self.liquid_indices = liquid_indices
        self.wall_height = wall_height
        self.type_model = type_model
        self.width_masspain = width_masspain
        self.output_dir = output_dir

        # Set default binning parameters if not provided
        if binning_params is None:
            self.binning_params = {
                'xi_0': 0, 'xi_f': 100.0, 'nbins_xi': 50,
                'zi_0': 0.0, 'zi_f': 100.0, 'nbins_zi': 50
            }
        else:
            self.binning_params = binning_params

        self._initialize_grid()
        os.makedirs(self.output_dir, exist_ok=True)
        matplotlib.use('Agg')

    def _initialize_grid(self):
        """Initialize the spatial grid based on binning parameters."""
        self.xi = np.linspace(
            self.binning_params['xi_0'],
            self.binning_params['xi_f'],
            self.binning_params['nbins_xi']
        )
        self.zi = np.linspace(
            self.binning_params['zi_0'],
            self.binning_params['zi_f'],
            self.binning_params['nbins_zi']
        )
        self.dxi = self.xi[1] - self.xi[0]
        self.dzi = self.zi[1] - self.zi[0]
        self.xi_cc = 0.5 * (self.xi[1:] + self.xi[:-1])
        self.zi_cc = 0.5 * (self.zi[1:] + self.zi[:-1])

    def binning(self, xi_par, zi_par, len_frames, type_model=None, width_masspain=None):
        """
        Bin particle data into a 2D grid to compute the density field.
        """
        if type_model is None:
            type_model = self.type_model
        if width_masspain is None:
            width_masspain = self.width_masspain

        rho_cc = np.zeros((len(self.xi_cc), len(self.zi_cc)))
        xi_par_0, zi_par_0 = np.copy(xi_par), np.copy(zi_par)

        for i in range(len(self.xi_cc)):
            if type_model == "masspain":
                dV = 2 * width_masspain * self.dxi * self.dzi
            elif type_model == "spherical":
                dV = 2 * np.pi * (self.xi_cc[i]) * self.dxi * self.dzi
            else:
                raise ValueError(f"Unknown model type: {type_model}. Use 'masspain' or 'spherical'.")

            for j in range(len(self.zi_cc)):
                where = (
                    (xi_par_0 > self.xi[i]) &
                    (xi_par_0 < self.xi[i + 1]) &
                    (zi_par_0 > self.zi[j]) &
                    (zi_par_0 < self.zi[j + 1])
                )
                count_i = np.sum(where)
                rho_cc[i, j] = count_i / dV
                # Optionally remove counted particles for efficiency
                in_this_bin_indx = np.nonzero(where)
                xi_par_0 = np.delete(xi_par_0, in_this_bin_indx)
                zi_par_0 = np.delete(zi_par_0, in_this_bin_indx)

        rho_cc /= len_frames
        return rho_cc

    def get_liquid_coords(self, frame_list):
        """
        Get cylindrical coordinates of liquid particles for a list of frames.
        This method is parser-agnostic: it expects the parser to return positions.
        """
        xi_par, zi_par = [], []
        for frame in frame_list:
            coords = self.parser.parse(num_frame=frame, indices=self.liquid_indices)
            # Convert to cylindrical coordinates (assume x, y, z)
            # If your system is 2D, adjust accordingly
            r = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2)
            xi_par.append(r)
            zi_par.append(coords[:, 2])
        xi_par = np.concatenate(xi_par)
        zi_par = np.concatenate(zi_par)
        return xi_par, zi_par, len(frame_list)

    def process_batch(self, frame_list, model=None, batch_index=None):
        """
        Process a batch of frames to calculate contact angle.
        """
        xi_par, zi_par, len_frames = self.get_liquid_coords(frame_list)
        particles_number = len(xi_par) / len_frames
        rho_cc = self.binning(xi_par, zi_par, len_frames)
        if model is None:
            model = HyperbolicTangentModel()
        msh_zi_cc_grid, msh_xi_cc_grid = np.meshgrid(self.zi_cc, self.xi_cc)
        msh_zi_cc = msh_zi_cc_grid.reshape((len(self.xi_cc) * len(self.zi_cc)), order="F")
        msh_xi_cc = msh_xi_cc_grid.reshape((len(self.xi_cc) * len(self.zi_cc)), order="F")
        msh_rho_cc = rho_cc.reshape((len(self.xi_cc) * len(self.zi_cc)), order="F")
        x_data = (msh_xi_cc, msh_zi_cc)
        model.fit(x_data, msh_rho_cc)
        contact_angle = model.compute_contact_angle(self.wall_height)
        circle_xi, circle_zi, wall_line_xi, wall_line_zi = model.compute_isoline()
        self.plot_density_with_isoline(
            self.xi_cc, self.zi_cc, rho_cc,
            circle_xi, circle_zi, wall_line_xi, wall_line_zi,
            batch_index
        )
        self.save_logfile(
            particles_number, model.get_parameter_strings(), contact_angle,
            self.xi_cc, self.zi_cc, rho_cc,
            batch_index
        )
        return contact_angle, model

    def plot_density_with_isoline(self, xi_cc, zi_cc, rho_cc, circle_xi, circle_zi,
                                 wall_line_xi, wall_line_zi, batch_index=None,
                                 clevels=20, scale=0.75, close=True):
        name = f"bin_plot_batch_{batch_index}.png" if batch_index is not None else "bin_plot.png"
        fig = plt.figure(dpi=300, figsize=(4 * scale, 3 * scale))
        plt.contourf(xi_cc, zi_cc, np.transpose(rho_cc), levels=clevels, cmap="jet")
        plt.colorbar()
        plt.plot(circle_xi, circle_zi, "--", color="black")
        plt.plot(wall_line_xi, wall_line_zi, "--", color="black")
        plt.savefig(os.path.join(self.output_dir, name))
        if close:
            plt.close()

    def save_logfile(self, particles_number, param_strings, theta, xi_cc, zi_cc, rho_cc, batch_index=None):
        batch_str = f"_batch_{batch_index}" if batch_index is not None else ""
        with open(os.path.join(self.output_dir, f"log_data{batch_str}.txt"), 'w') as f:
            f.write("Simulation parameters:\n")
            f.write(f"reduced_particles_number:{particles_number}\n")
            f.write(f"model_type:{self.type_model}\n")
            if self.type_model == "masspain":
                f.write(f"width_masspain:{self.width_masspain}\n")
            f.write("Fitted parameters:\n")
            for param in param_strings:
                f.write(param)
            f.write("\n\nContact angle:{}".format(theta))
        msh_zi_cc_grid, msh_xi_cc_grid = np.meshgrid(zi_cc, xi_cc)
        msh_zi_cc = msh_zi_cc_grid.reshape((len(xi_cc) * len(zi_cc)), order="F")
        msh_xi_cc = msh_xi_cc_grid.reshape((len(xi_cc) * len(zi_cc)), order="F")
        msh_rho_cc = rho_cc.reshape((len(xi_cc) * len(zi_cc)), order="F")
        CSV = np.c_[msh_xi_cc, msh_zi_cc, msh_rho_cc]
        np.savetxt(
            os.path.join(self.output_dir, f"rho_field{batch_str}.csv"),
            CSV,
            delimiter=",",
            header=f"x_{len(xi_cc)},y_{len(zi_cc)},rho_{len(xi_cc) * len(zi_cc)}"
        )

    def process_all_batches(self, batch_size=100, save_angles=True):
        frames_tot = self.parser.frame_tot()
        angles = []
        for batch_index, start_frame in enumerate(range(0, frames_tot, batch_size)):
            frame_list = list(range(start_frame, min(start_frame + batch_size, frames_tot)))
            angle, _ = self.process_batch(frame_list, batch_index=batch_index + 1)
            angles.append(angle)
        if save_angles:
            np.save(os.path.join(self.output_dir, f'all_angles_{self.type_model}.npy'), np.array(angles))
        print("List of contact angles by batch:", angles)
        return angles