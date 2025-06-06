import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import glob
import os
matplotlib.use('Agg')
class SurfacePlotter:
    def __init__(self, directory='.'):
        """
        Initialize the SurfacePlotter with a directory containing the .npy files.
        """
        self.directory = directory

    def plot_surface_points(self, surface_file, popt_file, output_filename, limit_med=9.5):
        """
        Plot the surface points and highlight points above `limit_med` in black.
        """
        # Load surface data
        surface_data = np.load(surface_file)

        # Load popt data
        popt = np.load(popt_file)
        Xs, Ys, R = popt[1]

        # Plot settings
        fig, ax = plt.subplots(figsize=(10, 8))

        for i in range(surface_data.shape[0]):
            # Extract surface points for the current frame
            surf = surface_data[i]
            X_data = surf[:, 0]
            Y_data = surf[:, 1]

            # Highlight surface points above limit_med
            surf_sup = surf[(surf[:, 1] > limit_med)]
            sup_X = surf_sup[:, 0]
            sup_Y = surf_sup[:, 1]

            # Plot the surface points
            ax.plot(X_data, Y_data, label=f'Surface {i + 1}', color='blue', linewidth=0.8)

            # Plot the highlighted surface points above limit_med
            ax.scatter(sup_X, sup_Y, color='black', marker='X', label=f'Surface > {limit_med} (Frame {i + 1})')

        # Plot the circle using popt parameters
        circle = plt.Circle((Xs, Ys), R, color='red', fill=False, linestyle='dashed', linewidth=2)
        ax.add_artist(circle)

        # Styling
        ax.set_title("Surface Points", fontsize=14)
        ax.set_xlabel("X", fontsize=12)
        ax.set_ylabel("Y", fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True)
        ax.axis('equal')

        # Save and show the plot
        plt.savefig(output_filename, format='png')
        
    def process_files(self):
        """
        Process all surface and popt files in the directory.
        """
        # Get all .npy files in the directory
        npy_files = glob.glob(os.path.join(self.directory, '*.npy'))

        # Filter to get only surface files and their corresponding popt files
        surface_files = [file for file in npy_files if 'surfaces' in file]
        popt_files = [file for file in npy_files if 'popts' in file]

        # Sort the files to ensure they match correctly
        surface_files.sort()
        popt_files.sort()

        # Iterate through each surface file and its corresponding popt file
        for surface_file, popt_file in zip(surface_files, popt_files):
            # Define an output filename for the plot
            output_filename = f"plot_{os.path.basename(surface_file).replace('.npy', '.png')}"
            self.plot_surface_points(surface_file, popt_file, output_filename)

# Example usage:
# plotter = SurfacePlotter(directory='.')
# plotter.process_files()
