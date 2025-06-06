import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from .surface_definition import HyperbolicTangentModel
import copy
     
class ContactAngleAnalyzer:
    """Class for analyzing contact angles in molecular dynamics simulations."""

    def __init__(self, parser, wall_height, type_model="spherical", width_masspain=21,
                 binning_params=None, output_dir="output_analysis/"):
        """
        Initialize the contact angle analyzer.
        
        Parameters:
        parser: DumpParser object for reading trajectory data
        wall_height: height of the wall surface
        type_model: type of model for volume calculation ("spherical" or "masspain")
        width_masspain: width parameter for masspain model
        binning_params: dict with binning parameters (optional)
        output_dir: directory for output files
        """
        self.parser = parser
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

        # Initialize grid
        self._initialize_grid()

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Use 'Agg' backend for matplotlib to save figures without displaying them
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

        # Cell centers
        self.xi_cc = 0.5 * (self.xi[1:] + self.xi[:-1])
        self.zi_cc = 0.5 * (self.zi[1:] + self.zi[:-1])

    def binning(self, xi_par, zi_par, len_frames, type_model=None, width_masspain=None):
        """
        Bin particle data into a 2D grid to compute the density field.
        
        Parameters:
        xi_par: array of xi coordinates of particles
        zi_par: array of zi coordinates of particles
        len_frames: number of frames used for averaging
        type_model: type of model for volume calculation (overrides instance setting if provided)
        width_masspain: width parameter for masspain model (overrides instance setting if provided)
        
        Returns:
        numpy.ndarray: 2D density field
        """
        # Use instance values as defaults if not provided
        if type_model is None:
            type_model = self.type_model
        if width_masspain is None:
            width_masspain = self.width_masspain

        print(f"Binning with model: {type_model} ...")

        # Create density grid
        rho_cc = np.zeros((len(self.xi_cc), len(self.zi_cc)))

        # Make copies to avoid modifying original data
        xi_par_0, zi_par_0 = copy.deepcopy(xi_par), copy.deepcopy(zi_par)

        for i in range(len(self.xi_cc)):
            if i % 10 == 0:
                print(f"Advancement: {100 * i / (len(self.xi_cc) - 1):.2f}%")

            # Calculate volume element
            if type_model == "masspain":
                dV = 2 * width_masspain * self.dxi * self.dzi
            elif type_model == "spherical":
                dV = 2 * np.pi * (self.xi_cc[i]) * self.dxi * self.dzi
            else:
                raise ValueError(f"Unknown model type: {type_model}. Use 'masspain' or 'spherical'.")

            for j in range(len(self.zi_cc)):
                # Find particles in this bin
                where = (
                    (xi_par_0 > self.xi[i]) * 
                    (xi_par_0 < self.xi[i + 1]) *
                    (zi_par_0 > self.zi[j]) * 
                    (zi_par_0 < self.zi[j + 1])
                )

                # Count particles and calculate density
                count_i = where.sum()
                rho_cc[i, j] = count_i / dV

                # Remove counted particles for efficiency
                in_this_bin_indx = np.nonzero(where)
                xi_par_0 = np.delete(xi_par_0, in_this_bin_indx)
                zi_par_0 = np.delete(zi_par_0, in_this_bin_indx)

        # Normalize by number of frames
        rho_cc /= len_frames

        return rho_cc

    def plot_density_with_isoline(self, xi_cc, zi_cc, rho_cc, circle_xi, circle_zi,
                                 wall_line_xi, wall_line_zi, batch_index=None,
                                 clevels=20, scale=0.75, close=True):
        """
        Plot the 2D density field and iso-surface lines.
        
        Parameters:
        xi_cc, zi_cc: Grid cell centers
        rho_cc: 2D density field
        circle_xi, circle_zi: Coordinates of the circle isoline
        wall_line_xi, wall_line_zi: Coordinates of the wall line
        batch_index: Index of the current batch (for file naming)
        clevels: Number of contour levels
        scale: Scale factor for figure size
        close: Whether to close the figure after saving
        """
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
        """
        Save the simulation parameters, fitted parameters, contact angle, and density field.
        
        Parameters:
        particles_number: Number of particles
        param_strings: List of parameter strings
        theta: Contact angle
        xi_cc, zi_cc: Grid cell centers
        rho_cc: 2D density field
        batch_index: Index of the current batch (for file naming)
        """
        batch_str = f"_batch_{batch_index}" if batch_index is not None else ""

        # Save log file
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

        # Save density field as CSV
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

    def process_batch(self, frame_list, model=None, batch_index=None):
        """
        Process a batch of frames to calculate contact angle.
        
        Parameters:
        frame_list: List of frame indices to process
        model: Surface model (default: HyperbolicTangentModel)
        batch_index: Index of the current batch (for file naming)
        
        Returns:
        tuple: (contact_angle, model)
        """

        # Get particle coordinates
        xi_par, zi_par, len_frames = self.parser.return_cylindrical_coord_pars(
            frame_list,
            type_model=self.type_model
        )

        # Calculate average number of particles
        particles_number = len(xi_par) / len_frames
        print(f"\nNumber of fluid particles in batch{' ' + str(batch_index) if batch_index is not None else ''}:\t{particles_number}\n")

        # Calculate density field
        rho_cc = self.binning(xi_par, zi_par, len_frames)

        # Initialize model if not provided
        if model is None:
            model = HyperbolicTangentModel()

        # Prepare data for fitting
        msh_zi_cc_grid, msh_xi_cc_grid = np.meshgrid(self.zi_cc, self.xi_cc)
        msh_zi_cc = msh_zi_cc_grid.reshape((len(self.xi_cc) * len(self.zi_cc)), order="F")
        msh_xi_cc = msh_xi_cc_grid.reshape((len(self.xi_cc) * len(self.zi_cc)), order="F")
        msh_rho_cc = rho_cc.reshape((len(self.xi_cc) * len(self.zi_cc)), order="F")

        # Fit model
        x_data = (msh_xi_cc, msh_zi_cc)
        model.fit(x_data, msh_rho_cc)

        # Get parameters and compute contact angle
        param_strings = model.get_parameter_strings()
        print("\nFitted parameters for batch{}:".format(f" {batch_index}" if batch_index is not None else ""))
        print("".join(param_strings))

        contact_angle = model.compute_contact_angle(self.wall_height)
        print(f"Contact angle for batch{' ' + str(batch_index) if batch_index is not None else ''}:\t{contact_angle}")

        # Compute iso-surface
        circle_xi, circle_zi, wall_line_xi, wall_line_zi = model.compute_isoline()

        # Plot and save results
        self.plot_density_with_isoline(
            self.xi_cc, self.zi_cc, rho_cc,
            circle_xi, circle_zi, wall_line_xi, wall_line_zi,
            batch_index
        )

        self.save_logfile(
            particles_number, param_strings, contact_angle,
            self.xi_cc, self.zi_cc, rho_cc,
            batch_index
        )

        return contact_angle, model

    def process_all_batches(self, batch_size=100, save_angles=True):
        """
        Process all frames in batches.
        
        Parameters:
        batch_size: Number of frames per batch
        save_angles: Whether to save angles to a numpy file
        
        Returns:
        list: List of contact angles
        """
        # Get total number of frames
        frames_tot = self.parser.frame_tot()
        print(f"Total frames: {frames_tot}")

        angles = []

        # Process each batch
        for batch_index, start_frame in enumerate(range(0, frames_tot, batch_size)):
            frame_list = list(range(start_frame, min(start_frame + batch_size, frames_tot)))

            # Process batch and get contact angle
            angle, _ = self.process_batch(frame_list, batch_index=batch_index + 1)
            angles.append(angle)

        # Save angles if requested
        if save_angles:
            np.save(os.path.join(self.output_dir, f'all_angles_{self.type_model}.npy'), np.array(angles))

        print("List of contact angles by batch:", angles)
        return angles


# # Example usage in a script
# if __name__ == "__main__":
#     from hydro_angle_analyzer import DumpParser

#     # Input parameters
#     in_dir = "/home/gtaillandier/Documents/contact_angle_lammps/edocolad/lib_python/small_test"
#     in_file = "traj_10_3_330w_nve_4k_reajust.lammpstrj"
#     in_path = os.path.join(in_dir, in_file)
#     out_dir = "/home/gtaillandier/Documents/contact_angle_lammps/edocolad/lib_python/small_test"

#     # Binning parameters
#     binning_params = {
#         'xi_0': 0, 'xi_f': 100.0, 'nbins_xi': 50,
#         'zi_0': 0.0, 'zi_f': 100.0, 'nbins_zi': 50
#     }

#     # Wall height and batch size
#     wall_height = 4.89
#     batch_size = 100
#     type_model = "masspain"  # or "spherical"
#     width_masspain = 21  # only used for masspain model

#     # Initialize parser and analyzer
#     parser = DumpParser(in_path)
#     analyzer = ContactAngleAnalyzer(
#         parser=parser,
#         wall_height=wall_height,
#         type_model=type_model,
#         width_masspain=width_masspain,
#         binning_params=binning_params,
#         output_dir=out_dir
#     )

#     # Process all batches
#     angles = analyzer.process_all_batches(batch_size=batch_size)

#     # Print results
#     print("Analysis complete!")
#     print(f"Average contact angle: {np.mean(angles):.2f} ± {np.std(angles):.2f}")
