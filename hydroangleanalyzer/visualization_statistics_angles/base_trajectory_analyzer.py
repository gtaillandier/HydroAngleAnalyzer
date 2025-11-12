import os
import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class BaseTrajectoryAnalyzer(ABC):
    def __init__(self, directories, time_unit="ps"):
        """
        Initialize the analyzer with a list of directory paths.

        Parameters
        ----------
        directories : list of str
            List of directory paths containing analysis results.
        time_unit : str, optional
            Time unit for the x-axis (e.g., "ps", "ns", "fs").
        """
        self.directories = directories
        self.data = {}
        self.time_unit = time_unit
        self._initialize_data_structure()

    
    @abstractmethod
    def _initialize_data_structure(self):
        """Initialize the data dictionary structure for each directory."""
        pass
    
    @abstractmethod
    def read_data(self):
        """Read and parse data from files in each directory."""
        pass
    
    @abstractmethod
    def get_surface_areas(self, directory):
        """
        Get surface areas for a given directory.
        
        Parameters
        ----------
        directory : str
            Directory path.
            
        Returns
        -------
        numpy.ndarray
            Array of surface area values.
        """
        pass
    
    @abstractmethod
    def get_contact_angles(self, directory):
        """
        Get contact angles for a given directory.
        
        Parameters
        ----------
        directory : str
            Directory path.
            
        Returns
        -------
        numpy.ndarray
            Array of contact angle values.
        """
        pass
    
    @abstractmethod
    def get_method_name(self):
        """
        Return the name of this analysis method.
        
        Returns
        -------
        str
            Method name for labels and titles.
        """
        pass
    
    def compute_statistics(self, directory):
        """
        Compute mean surface area, mean angle, and standard error.
        
        Parameters
        ----------
        directory : str
            Directory path.
            
        Returns
        -------
        tuple
            (x_value, y_value, y_error) where:
            - x_value: 1/sqrt(mean_surface_area)
            - y_value: mean contact angle
            - y_error: standard error of the mean
        """
        surface_areas = self.get_surface_areas(directory)
        contact_angles = self.get_contact_angles(directory)
        
        x = 1 / np.sqrt(np.mean(surface_areas))
        y = np.mean(contact_angles)
        yerr = np.std(contact_angles) / np.sqrt(len(contact_angles))
        
        return x, y, yerr
    
    def get_clean_label(self, directory):
        """
        Generate a clean label from directory name.
        
        Parameters
        ----------
        directory : str
            Directory path.
            
        Returns
        -------
        str
            Cleaned directory name for plotting.
        """
        return (directory.replace("_reduce_sliced", "")
                        .replace("_reduce_binned", "")
                        .replace("result_dump_", ""))
    
    def analyze(self, output_filename="output_stats.txt"):
        """
        Analyze and save statistics for each directory to an output file.

        Args:
            output_filename (str): Name of the output file (default: "output_stats.txt").
        """
        self.read_data()
        for directory in self.directories:
            output_path = f"{directory}/{output_filename}"  # Custom or default output filename

            with open(output_path, 'w') as f:
                f.write(f"Directory: {directory}\n")
                f.write(f"Method: {self.get_method_name()}\n")
                f.write(f"Mean Surface Area: {np.mean(self.get_surface_areas(directory)):.4f}\n")
                f.write(f"Mean Contact Angle: {np.mean(self.get_contact_angles(directory)):.4f}°\n")

            print(f"Analysis saved to: {output_path}")

    def plot_mean_angle_vs_surface(self, labels=None, colors=None,save_path=None):
        """
        Generate a professional academic plot comparing mean angle vs surface area scaling.
        If no analysis output is found, run the analysis first.

        Parameters
        ----------
        labels : list of str, optional
            Labels for each dataset. If None, directory names are used.
        colors : list of str, optional
            Custom colors for each dataset.
        save_path : str, optional
            Path to save the figure.
        """
        # Check if analysis output files exist; if not, run analysis
        for directory in self.directories:
            output_file = f"{directory}/output_stats.txt"
            if not os.path.exists(output_file):
                print(f"No analysis found for {directory}. Running analysis...")
                self.analyze()  # Run analysis to generate output files
                break  # Only need to run once

        # Read data if not already loaded
        if not hasattr(self, 'data') or not self.data:
            self.read_data()

        # Set up plot parameters
        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 13,
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "legend.fontsize": 12,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "axes.linewidth": 1.0,
            "errorbar.capsize": 3,
        })

        # Create the plot
        fig, ax = plt.subplots(figsize=(7, 4.5))

        # Set default labels and colors if not provided
        if labels is None:
            labels = [self.get_clean_label(d) for d in self.directories]
        if colors is None:
            colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(self.directories)))

        # Collect data for plotting
        xvals, yvals = [], []
        for d, label, color in zip(self.directories, labels, colors):
            # Read data from the analysis output file
            output_file = f"{d}/output_stats.txt"
            with open(output_file, 'r') as f:
                lines = f.readlines()
                mean_surface_area = float(lines[2].split(": ")[1].strip())
                mean_contact_angle = float(lines[3].split(": ")[1].strip())

            # Use the data for plotting
            x = 1 / np.sqrt(mean_surface_area)  # Example transformation
            y = mean_contact_angle
            yerr = 0.5  # Placeholder for error; adjust as needed

            ax.errorbar(
                x, y, yerr=yerr, fmt='o', color=color,
                markersize=6, capsize=3, lw=1.2
            )
            ax.annotate(
                label, xy=(x, y), xytext=(5, 5), textcoords="offset points",
                ha="left", va="center", fontsize=6, color="black"
            )
            xvals.append(x)
            yvals.append(y)

        # Linear fit
        xvals, yvals = np.array(xvals), np.array(yvals)
        coeffs = np.polyfit(xvals, yvals, 1)
        fit_line = np.poly1d(coeffs)
        x_fit = np.linspace(0, max(xvals) * 1.1, 100)
        ax.plot(x_fit, fit_line(x_fit), "--", color="gray", lw=1.5,
                label=f"Linear Fit (y = {fit_line(0):.2f}°)")

        # Set plot labels and title
        ax.set_xlabel(r"$1 / \sqrt{\text{Surface Area}}$")
        ax.set_ylabel("Mean Angle (°)")
        ax.set_title(f"{self.get_method_name()} - Mean Angle vs Surface Area", pad=10)
        ax.legend(frameon=False, loc="upper left")
        ax.grid(False)
        ax.set_xlim(left=-0.001)
        ax.set_ylim(bottom=min(yvals) - 2, top=max(yvals) + 2)
        plt.tight_layout()

        # Save the plot if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=400, bbox_inches="tight")
        plt.close()