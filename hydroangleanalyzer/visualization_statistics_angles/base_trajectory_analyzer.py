import os
import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class BaseTrajectoryAnalyzer(ABC):
    """
    Abstract base class for trajectory analysis methods.
    Provides common interface and plotting utilities.
    """
    
    def __init__(self, directories):
        """
        Initialize the analyzer with a list of directory paths.
        
        Parameters
        ----------
        directories : list of str
            List of directory paths containing analysis results.
        """
        self.directories = directories
        self.data = {}
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
    
    def analyze(self):
        """Analyze and print statistics for each directory."""
        self.read_data()
        for directory in self.directories:
            print(f"Directory: {directory}")
            print(f"  Method: {self.get_method_name()}")
            print(f"  Mean Surface Area: {np.mean(self.get_surface_areas(directory)):.4f}")
            print(f"  Mean Contact Angle: {np.mean(self.get_contact_angles(directory)):.4f}°")
            print()
    
    def plot_mean_angle_vs_surface(self, labels=None, colors=None, markers=None, save_path=None):
        """
        Generate a professional academic plot comparing mean angle vs surface area scaling.
        
        Parameters
        ----------
        labels : list of str, optional
            Labels for each dataset. If None, directory names are used.
        colors : list of str, optional
            Custom colors for each dataset.
        markers : list of str, optional
            Custom marker styles for each dataset.
        save_path : str, optional
            Path to save the figure.
        """
        if not hasattr(self, 'data') or not self.data:
            self.read_data()
        
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
        
        fig, ax = plt.subplots(figsize=(7, 4.5))
        
        if labels is None:
            labels = [self.get_clean_label(d) for d in self.directories]
        if colors is None:
            colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(self.directories)))
        if markers is None:
            markers = ["o", "s", "^", "D", "v", "p", "h", "X"][:len(self.directories)]
        
        xvals, yvals = [], []
        for d, label, color, marker in zip(self.directories, labels, colors, markers):
            x, y, yerr = self.compute_statistics(d)
            ax.errorbar(
                x, y, yerr=yerr, fmt=marker, color=color,
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
        
        ax.set_xlabel(r"$1 / \sqrt{\text{Surface Area}}$")
        ax.set_ylabel("Mean Angle (°)")
        ax.set_title(f"{self.get_method_name()} - Mean Angle vs Surface Area", pad=10)
        ax.legend(frameon=False, loc="upper left")
        ax.grid(False)
        ax.set_xlim(left=-0.001)
        ax.set_ylim(bottom=min(yvals) - 2, top=max(yvals) + 2)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=400, bbox_inches="tight")
        plt.close()