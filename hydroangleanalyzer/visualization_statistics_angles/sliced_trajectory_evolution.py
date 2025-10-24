import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib import cm

class Sliced_Trajectory_Analyzer:
    def __init__(self, directories):
        """
        Initialize the TrajectoryAnalyzer with a list of directory paths.
        """
        self.directories = directories
        self.data = {}  # Dictionary to store data for each directory
        for directory in directories:
            self.data[directory] = {
                "surfaces_files": [],
                "popts_files": [],
                "alfas_files": [],
                "mean_surface_areas": [],
                "all_alfas": [],
                "median_alfas": [],
                "std_alfas": [],
            }

    def calculate_polygon_area(self, points):
        """
        Calculate the area of a polygon given its vertices using the Shoelace formula.
        """
        x = points[:, 0]
        y = points[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        return area

    def mean_surface_frame(self, surfaces_file):
        """
        Calculate the mean surface area for a given surfaces file.
        """
        surfaces = np.load(surfaces_file, allow_pickle=True)
        all_surf = [self.calculate_polygon_area(surface) for surface in surfaces]
        return np.mean(np.array(all_surf))

    def load_files(self):
        """
        Load and sort all relevant files from each directory.
        """
        for directory in self.directories:
            self.data[directory]["surfaces_files"] = sorted(glob.glob(os.path.join(directory, "surfacesframe*.npy")))
            self.data[directory]["popts_files"] = sorted(glob.glob(os.path.join(directory, "poptsframe*.npy")))
            self.data[directory]["alfas_files"] = sorted(glob.glob(os.path.join(directory, "alfasframe*.txt")))

            if not (len(self.data[directory]["surfaces_files"]) ==
                    len(self.data[directory]["popts_files"]) ==
                    len(self.data[directory]["alfas_files"])):
                raise ValueError(f"Mismatch in the number of files for directory: {directory}")


    def analyze(self):
        """
        Analyze each directory and log statistics.
        """
        self.load_files()
        for directory in self.directories:
            for surf_file, popt_file, alfa_file in zip(
                self.data[directory]["surfaces_files"],
                self.data[directory]["popts_files"],
                self.data[directory]["alfas_files"],
            ):
                num_frame = int(os.path.basename(surf_file).replace("surfacesframe", "").replace(".npy", ""))
                popts = np.load(popt_file)
                mean_surface_area = self.mean_surface_frame(surf_file)
                alfas = np.loadtxt(alfa_file)

                self.data[directory]["mean_surface_areas"].append(mean_surface_area)
                self.data[directory]["all_alfas"].append(alfas)
                self.data[directory]["median_alfas"].append(np.median(alfas))
                self.data[directory]["std_alfas"].append(np.std(alfas))

                print(f"Directory: {directory}, Frame {num_frame}:")
                print(f"  Mean Surface Area: {mean_surface_area}")
                print(f"  Median Alfas: {np.median(alfas)}")
                print(f"  Std Alfas: {np.std(alfas)}")

    def plot_alfas_evolution(self, save_path):
        """
        Plot the evolution of the angle (Alfas) for each slice across frames.
        """
        plt.figure(figsize=(12, 8))
        colormap = cm.get_cmap('viridis', len(self.all_alfas[0]))
        for slice_idx in range(len(self.all_alfas[0])):
            slice_alfas = [frame_alfas[slice_idx] for frame_alfas in self.all_alfas]
            plt.plot(slice_alfas, marker='o', linestyle='-', color=colormap(slice_idx), label=f'Slice {slice_idx}')

        plt.title("Evolution of the Angle (Alfas) for Each Slice Across Frames")
        plt.xlabel("Frame Number")
        plt.ylabel("Angle (Alfas)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=400, bbox_inches="tight")  # Save instead of show
        plt.close()  # Close the figure to free memory
    def plot_median_alfas_evolution(self, save_path):
        """
        Plot the evolution of the median angle (Alfas) with standard deviation.
        """
        plt.figure(figsize=(10, 6))
        frame_numbers = list(range(len(self.median_alfas)))
        plt.plot(frame_numbers, self.median_alfas, marker='o', linestyle='-', color='b', label='Median Angle')
        plt.fill_between(frame_numbers,
                        np.array(self.median_alfas) - np.array(self.std_alfas),
                        np.array(self.median_alfas) + np.array(self.std_alfas),
                        color='b', alpha=0.2, label='±1 Std Dev')

        plt.title("Evolution of the Median Angle (Alfas) with Standard Deviation")
        plt.xlabel("Frame Number")
        plt.ylabel("Angle (Alfas)")
        plt.legend()
        plt.grid(False)
        plt.savefig(save_path, dpi=400, bbox_inches="tight")  # Save instead of show
        plt.close()  # Close the figure to free memory
    def plot_comparison_median_alfas(self, save_path):
        """
        Plot a comparison of the median angle (Alfas) with standard deviation for all directories.
        """
        plt.figure(figsize=(12, 8))
        frame_numbers = list(range(len(self.data[self.directories[0]]["median_alfas"])))

        for directory in self.directories:
            median_alfas = self.data[directory]["median_alfas"]
            std_alfas = self.data[directory]["std_alfas"]

            plt.plot(frame_numbers, median_alfas, marker='o', linestyle='-', label=f'Median Angle ({directory})')
            plt.fill_between(
                frame_numbers,
                np.array(median_alfas) - np.array(std_alfas),
                np.array(median_alfas) + np.array(std_alfas),
                alpha=0.2,
                label=f'±1 Std Dev ({directory})',
            )

        plt.title("Comparison of Median Angle (Alfas) with Standard Deviation")
        plt.xlabel("Frame Number")
        plt.ylabel("Angle (Alfas)")
        plt.legend()
        plt.grid(False)
        plt.savefig(save_path, dpi=400, bbox_inches="tight")
        plt.close()

