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
                alpha=0.2
            )

        plt.title("Comparison of Median Angle (Alfas) with Standard Deviation")
        plt.xlabel("Frame Number")
        plt.ylabel("Angle (Alfas)")
        plt.legend()
        plt.grid(False)
        plt.savefig(save_path, dpi=400, bbox_inches="tight")
        plt.close()
    def plot_mean_angle_vs_surface(self, labels=None, colors=None, markers=None, save_path=None):
        """
        Generate a professional academic plot comparing mean angle vs surface area scaling.

        Parameters
        ----------
        labels : list of str, optional
            Labels for each dataset (for the legend). If None, directory names are used.
        colors : list of str, optional
            Custom colors for each dataset. Default uses Matplotlib's color cycle.
        markers : list of str, optional
            Custom marker styles for each dataset.
        save_path : str, optional
            If provided, saves the figure to this path (e.g. "angle_vs_surface.png").
        """
        # --- Initialize plotting setup ---
        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 13,
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "legend.fontsize": 12,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "axes.linewidth": 1.0,
            "errorbar.capsize": 3
        })
        fig, ax = plt.subplots(figsize=(7, 4.5))

        # --- Defaults for styling ---
        if labels is None:
            labels = [d.replace("_reduce_sliced", "").replace("result_dump_", "") for d in self.directories]
        if colors is None:
            colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(self.directories)))
        if markers is None:
            markers = ["o", "s", "^", "D", "v", "p", "h", "X"][:len(self.directories)]

        # --- Collect and plot data ---
        xvals, yvals = [], []
        for d, label, color, marker in zip(self.directories, labels, colors, markers):
            mean_surface_areas = self.data[d]["mean_surface_areas"]
            median_alfas = self.data[d]["median_alfas"]
            std_alfas = self.data[d]["std_alfas"]

            x = 1 / np.sqrt(np.mean(mean_surface_areas))
            y = np.mean(median_alfas)
            yerr = np.std(median_alfas) / np.sqrt(len(median_alfas))  # SEM estimate

            ax.errorbar(
            x, y, yerr=yerr, fmt=marker, color=color,
            markersize=6, capsize=3, lw=1.2
            )
            ax.annotate(
                label,
                xy=(x, y),
                xytext=(5, 5),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=6,
                color="black"
                )
            xvals.append(x)
            yvals.append(y)

        # --- Fit and plot linear regression ---
        xvals, yvals = np.array(xvals), np.array(yvals)
        coeffs = np.polyfit(xvals, yvals, 1)
        fit_line = np.poly1d(coeffs)
        x_fit = np.linspace(0, max(xvals) * 1.1, 100)
        ax.plot(x_fit, fit_line(x_fit), "--", color="gray", lw=1.5, label=f"Linear Fit (y = {fit_line(0):.2f}°)")

        # --- Axis labels and title ---
        ax.set_xlabel(r"$1 / \sqrt{\text{Surface Area of Water Molecules}}$")
        ax.set_ylabel("Mean Angle (°)")
        ax.set_title("Cylindrical Fit Convergence with Surface Area", pad=10)

        # --- Legend, grid, limits ---
        ax.legend(frameon=False, loc="upper left")
        ax.grid(False)
        ax.set_xlim(left=-0.001)
        ax.set_ylim(bottom=min(yvals) - 2, top=max(yvals) + 2)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=400, bbox_inches="tight")
        plt.close()
