import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib import cm

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
        self.time_steps = time_steps if time_steps else {d: 1.0 for d in directories}
        self.time_unit = time_unit
        super().__init__(directories, time_unit=time_unit)

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
                "mean_alfas": [],
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
                self.data[directory]["mean_alfas"].append(np.mean(alfas))
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
                self.data[directory]["mean_alfas"].append(np.mean(alfas))
                self.data[directory]["std_alfas"].append(np.std(alfas))
    def plot_median_alfas_evolution(self, save_path, labels=None):
        """
        Plot the evolution of the median angle (Alfas) with standard deviation for all directories.
        Aligns trajectories by truncating to the shortest.

        Parameters
        ----------
        save_path : str
            Path to save the plot.
        labels : dict, optional
            Dictionary mapping directory to a custom label for plotting.
            If None, directory basename is used.
        """
        if not self.data[self.directories[0]]["median_alfas"]:
            self.analyze_alfas_only()

        # Use provided labels or fall back to directory basename
        plot_labels = labels if labels else {d: os.path.basename(d) for d in self.directories}

        # Find the minimum number of frames across all directories
        min_frames = min(len(self.data[d]["median_alfas"]) for d in self.directories)
        plt.figure(figsize=(10, 6))
        colors = plt.cm.tab20(np.linspace(0, 1, len(self.directories)))

        for i, directory in enumerate(self.directories):
            median_alfas = self.data[directory]["median_alfas"][:min_frames]
            std_alfas = self.data[directory]["std_alfas"][:min_frames]
            time_step = self.data[directory]["time_step"]
            time_values = np.arange(min_frames) * time_step
            label = plot_labels.get(directory, os.path.basename(directory))

            plt.plot(
                time_values,
                median_alfas,
                linestyle='-',
                color=colors[i],
                label=f'Median Angle ({label})'
            )
            plt.fill_between(
                time_values,
                np.array(median_alfas) - np.array(std_alfas),
                np.array(median_alfas) + np.array(std_alfas),
                color=colors[i],
                alpha=0.2,
                label=f'±1 Std Dev ({label})'
            )

        plt.title("Evolution of the Median Angle (Alfas) with Standard Deviation")
        plt.xlabel(f"Time ({self.time_unit})")
        plt.ylabel("Angle (Alfas)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=400, bbox_inches="tight")
        plt.close()

    def plot_mean_alfas_evolution(self, save_path, labels=None):
        """
        Plot the evolution of the mean angle (Alfas) with standard deviation for all directories.
        Aligns trajectories by truncating to the shortest.

        Parameters
        ----------
        save_path : str
            Path to save the plot.
        labels : dict, optional
            Dictionary mapping directory to a custom label for plotting.
            If None, directory basename is used.
        """
        if not self.data[self.directories[0]]["mean_alfas"]:
            self.analyze_alfas_only()

        # Use provided labels or fall back to directory basename
        plot_labels = labels if labels else {d: os.path.basename(d) for d in self.directories}

        # Find the minimum number of frames across all directories
        min_frames = min(len(self.data[d]["mean_alfas"]) for d in self.directories)
        plt.figure(figsize=(10, 6))
        colors = plt.cm.tab20(np.linspace(0, 1, len(self.directories)))

        for i, directory in enumerate(self.directories):
            mean_alfas = self.data[directory]["mean_alfas"][:min_frames]
            std_alfas = self.data[directory]["std_alfas"][:min_frames]
            time_step = self.data[directory]["time_step"]
            time_values = np.arange(min_frames) * time_step
            label = plot_labels.get(directory, os.path.basename(directory))

            plt.plot(
                time_values,
                mean_alfas,
                linestyle='-',
                color=colors[i],
                label=f'Mean Angle ({label})'
            )
            plt.fill_between(
                time_values,
                np.array(mean_alfas) - np.array(std_alfas),
                np.array(mean_alfas) + np.array(std_alfas),
                color=colors[i],
                alpha=0.2,
                label=f'±1 Std Dev ({label})'
            )

        plt.title("Evolution of the Mean Angle (Alfas) with Standard Deviation")
        plt.xlabel(f"Time ({self.time_unit})")
        plt.ylabel("Angle (Alfas)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=400, bbox_inches="tight")
        plt.close()