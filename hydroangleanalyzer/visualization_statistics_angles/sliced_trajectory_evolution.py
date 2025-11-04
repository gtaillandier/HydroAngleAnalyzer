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
        self.data = {}
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
            self.data[directory]["surfaces_files"] = sorted(
                glob.glob(os.path.join(directory, "surfacesframe*.npy")),
                key=lambda x: int(os.path.basename(x).replace("surfacesframe", "").replace(".npy", ""))
            )
            self.data[directory]["popts_files"] = sorted(
                glob.glob(os.path.join(directory, "poptsframe*.npy")),
                key=lambda x: int(os.path.basename(x).replace("poptsframe", "").replace(".npy", ""))
            )
            self.data[directory]["alfas_files"] = sorted(
                glob.glob(os.path.join(directory, "alfasframe*.txt")),
                key=lambda x: int(os.path.basename(x).replace("alfasframe", "").replace(".txt", ""))
            )
            if not (len(self.data[directory]["surfaces_files"]) ==
                    len(self.data[directory]["popts_files"]) ==
                    len(self.data[directory]["alfas_files"])):
                raise ValueError(f"Mismatch in the number of files for directory: {directory}")

    def analyze_alfas_only(self):
        """
        Analyze only the alfas data (skip mean surface area calculation).
        """
        self.load_files()
        for directory in self.directories:
            for alfa_file in self.data[directory]["alfas_files"]:
                alfas = np.loadtxt(alfa_file)
                self.data[directory]["all_alfas"].append(alfas)
                self.data[directory]["median_alfas"].append(np.median(alfas))
                self.data[directory]["std_alfas"].append(np.std(alfas))

    def analyze_mean_angle_vs_surface(self):
        """
        Analyze only the data needed for the mean angle vs surface plot.
        """
        self.load_files()
        for directory in self.directories:
            for surf_file, alfa_file in zip(
                self.data[directory]["surfaces_files"],
                self.data[directory]["alfas_files"],
            ):
                mean_surface_area = self.mean_surface_frame(surf_file)
                alfas = np.loadtxt(alfa_file)
                self.data[directory]["mean_surface_areas"].append(mean_surface_area)
                self.data[directory]["median_alfas"].append(np.median(alfas))
                self.data[directory]["std_alfas"].append(np.std(alfas))

    def plot_median_alfas_evolution(self, save_path):
        """
        Plot the evolution of the median angle (Alfas) with standard deviation for all directories.
        Aligns trajectories by truncating to the shortest.
        """
        if not self.data[self.directories[0]]["median_alfas"]:
            self.analyze_alfas_only()

        # Find the minimum number of frames across all directories
        min_frames = min(len(self.data[d]["median_alfas"]) for d in self.directories)
        frame_numbers = list(range(min_frames))

        plt.figure(figsize=(10, 6))
        colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(self.directories)))

        for i, directory in enumerate(self.directories):
            median_alfas = self.data[directory]["median_alfas"][:min_frames]
            std_alfas = self.data[directory]["std_alfas"][:min_frames]

            plt.plot(
                frame_numbers,
                median_alfas,
                marker='o',
                linestyle='-',
                color=colors[i],
                label=f'Median Angle ({os.path.basename(directory)})'
            )
            plt.fill_between(
                frame_numbers,
                np.array(median_alfas) - np.array(std_alfas),
                np.array(median_alfas) + np.array(std_alfas),
                color=colors[i],
                alpha=0.2,
                label=f'±1 Std Dev ({os.path.basename(directory)})'
            )

        plt.title("Evolution of the Median Angle (Alfas) with Standard Deviation")
        plt.xlabel("Frame Number")
        plt.ylabel("Angle (Alfas)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=400, bbox_inches="tight")
        plt.close()


    def plot_mean_angle_vs_surface(self, labels=None, colors=None, markers=None, save_path=None):
        """
        Generate a professional academic plot comparing mean angle vs surface area scaling.
        Only analyzes the required data if not already done.
        """
        if not self.data[self.directories[0]]["mean_surface_areas"]:
            self.analyze_mean_angle_vs_surface()

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
        if labels is None:
            labels = [d.replace("_reduce_sliced", "").replace("result_dump_", "") for d in self.directories]
        if colors is None:
            colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(self.directories)))
        if markers is None:
            markers = ["o", "s", "^", "D", "v", "p", "h", "X"][:len(self.directories)]
        xvals, yvals = [], []
        for d, label, color, marker in zip(self.directories, labels, colors, markers):
            mean_surface_areas = self.data[d]["mean_surface_areas"]
            median_alfas = self.data[d]["median_alfas"]
            std_alfas = self.data[d]["std_alfas"]
            x = 1 / np.sqrt(np.mean(mean_surface_areas))
            y = np.mean(median_alfas)
            yerr = np.std(median_alfas) / np.sqrt(len(median_alfas))
            ax.errorbar(x, y, yerr=yerr, fmt=marker, color=color, markersize=6, capsize=3, lw=1.2)
            ax.annotate(label, xy=(x, y), xytext=(5, 5), textcoords="offset points", ha="left", va="center", fontsize=6, color="black")
            xvals.append(x)
            yvals.append(y)
        coeffs = np.polyfit(xvals, yvals, 1)
        fit_line = np.poly1d(coeffs)
        x_fit = np.linspace(0, max(xvals) * 1.1, 100)
        ax.plot(x_fit, fit_line(x_fit), "--", color="gray", lw=1.5, label=f"Linear Fit (y = {fit_line(0):.2f}°)")
        ax.set_xlabel(r"$1 / \sqrt{\text{Surface Area of Water Molecules}}$")
        ax.set_ylabel("Mean Angle (°)")
        ax.set_title("Cylindrical Fit Convergence with Surface Area", pad=10)
        ax.legend(frameon=False, loc="upper left")
        ax.grid(False)
        ax.set_xlim(left=-0.001)
        ax.set_ylim(bottom=min(yvals) - 2, top=max(yvals) + 2)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=400, bbox_inches="tight")
        plt.close()
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from .base_trajectory_analyzer import BaseTrajectoryAnalyzer

class SlicedTrajectoryAnalyzer(BaseTrajectoryAnalyzer):
    def __init__(self, directories, time_steps=None, time_unit="ps"):
        """
        Initialize the analyzer with a list of directory paths.

        Parameters
        ----------
        directories : list of str
            List of directory paths containing analysis results.
        time_steps : dict, optional
            Dictionary mapping directory to its time step.
            If None, defaults to 1.0 for all directories.
        time_unit : str, optional
            Time unit for the x-axis (e.g., "ps", "ns", "fs").
        """
        super().__init__(directories, time_unit=time_unit)
        self.time_unit = time_unit
        self.time_steps = time_steps if time_steps else {d: 1.0 for d in directories}

    def _initialize_data_structure(self):
        """Initialize data structure for sliced analysis."""
        for directory in self.directories:
            self.data[directory] = {
                "surfaces_files": [],
                "popts_files": [],
                "alfas_files": [],
                "mean_surface_areas": [],
                "all_alfas": [],
                "median_alfas": [],
                "std_alfas": [],
                "time_step": self.time_steps.get(directory, 1.0),
            }

    
    def get_method_name(self):
        """Return method name."""
        return "Sliced Analysis"
    
    @staticmethod
    def calculate_polygon_area(points):
        """
        Calculate the area of a polygon given its vertices using the Shoelace formula.
        
        Parameters
        ----------
        points : numpy.ndarray
            Array of shape (N, 2) containing polygon vertices.
            
        Returns
        -------
        float
            Area of the polygon.
        """
        x = points[:, 0]
        y = points[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        return area
    
    def mean_surface_frame(self, surfaces_file):
        """
        Calculate the mean surface area for a given surfaces file.
        
        Parameters
        ----------
        surfaces_file : str
            Path to the surfaces file.
            
        Returns
        -------
        float
            Mean surface area across all surfaces in the frame.
        """
        surfaces = np.load(surfaces_file, allow_pickle=True)
        all_surf = [self.calculate_polygon_area(surface) for surface in surfaces]
        return np.mean(np.array(all_surf))
    
    def load_files(self):
        """Load and sort all relevant files from each directory."""
        for directory in self.directories:
            self.data[directory]["surfaces_files"] = sorted(
                glob.glob(os.path.join(directory, "surfacesframe*.npy")),
                key=lambda x: int(os.path.basename(x).replace("surfacesframe", "").replace(".npy", ""))
            )
            self.data[directory]["popts_files"] = sorted(
                glob.glob(os.path.join(directory, "poptsframe*.npy")),
                key=lambda x: int(os.path.basename(x).replace("poptsframe", "").replace(".npy", ""))
            )
            self.data[directory]["alfas_files"] = sorted(
                glob.glob(os.path.join(directory, "alfasframe*.txt")),
                key=lambda x: int(os.path.basename(x).replace("alfasframe", "").replace(".txt", ""))
            )
            if not (len(self.data[directory]["surfaces_files"]) ==
                    len(self.data[directory]["popts_files"]) ==
                    len(self.data[directory]["alfas_files"])):
                raise ValueError(f"Mismatch in the number of files for directory: {directory}")
    
    def read_data(self):
        """Read and analyze data from files."""
        self.load_files()
        for directory in self.directories:
            for surf_file, alfa_file in zip(
                self.data[directory]["surfaces_files"],
                self.data[directory]["alfas_files"],
            ):
                mean_surface_area = self.mean_surface_frame(surf_file)
                alfas = np.loadtxt(alfa_file)
                self.data[directory]["mean_surface_areas"].append(mean_surface_area)
                self.data[directory]["median_alfas"].append(np.median(alfas))
                self.data[directory]["std_alfas"].append(np.std(alfas))
    
    def get_surface_areas(self, directory):
        """Get surface areas for a directory."""
        return np.array(self.data[directory]["mean_surface_areas"])
    
    def get_contact_angles(self, directory):
        """Get contact angles (median alfas) for a directory."""
        return np.array(self.data[directory]["median_alfas"])
    
    def analyze_alfas_only(self):
        """
        Analyze only the alfas data (skip mean surface area calculation).
        """
        self.load_files()
        for directory in self.directories:
            for alfa_file in self.data[directory]["alfas_files"]:
                alfas = np.loadtxt(alfa_file)
                self.data[directory]["all_alfas"].append(alfas)
                self.data[directory]["median_alfas"].append(np.median(alfas))
                self.data[directory]["std_alfas"].append(np.std(alfas))
    def plot_median_alfas_evolution(self, save_path):
        """
        Plot the evolution of the median angle (Alfas) with standard deviation for all directories.
        Aligns trajectories by truncating to the shortest.
        """
        if not self.data[self.directories[0]]["median_alfas"]:
            self.analyze_alfas_only()

        # Find the minimum number of frames across all directories
        min_frames = min(len(self.data[d]["median_alfas"]) for d in self.directories)

        plt.figure(figsize=(10, 6))
        colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(self.directories)))

        for i, directory in enumerate(self.directories):
            median_alfas = self.data[directory]["median_alfas"][:min_frames]
            std_alfas = self.data[directory]["std_alfas"][:min_frames]
            time_step = self.data[directory]["time_step"]
            time_values = np.arange(min_frames) * time_step

            plt.plot(
                time_values,
                median_alfas,
                marker='o',
                linestyle='-',
                color=colors[i],
                label=f'Median Angle ({os.path.basename(directory)})'
            )

            plt.fill_between(
                time_values,
                np.array(median_alfas) - np.array(std_alfas),
                np.array(median_alfas) + np.array(std_alfas),
                color=colors[i],
                alpha=0.2,
                label=f'±1 Std Dev ({os.path.basename(directory)})'
            )

        plt.title("Evolution of the Median Angle (Alfas) with Standard Deviation")
        plt.xlabel(f"Time ({self.time_unit})")
        plt.ylabel("Angle (Alfas)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=400, bbox_inches="tight")
        plt.close()