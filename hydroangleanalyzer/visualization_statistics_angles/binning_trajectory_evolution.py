import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

class Binning_Trajectory_Analyzer:
    def __init__(self, directories):
        """
        Initialize the Binning_Trajectory_Analyzer with a list of directory paths.

        Parameters
        ----------
        directories : list of str
            List of directory paths containing result_dump_* directories.
        """
        self.directories = directories
        self.data = {}
        for directory in directories:
            self.data[directory] = {
                "R_eq": [],
                "zi_c": [],
                "zi_0": [],
                "contact_angles": [],
                "surface_areas": [],
            }

    @staticmethod
    def circular_segment_area(R, z_center, z_cut):
        """
        Compute the area of a circular segment (cap) of a circle with radius R.
        Handles any cut position above or below the center.

        Parameters
        ----------
        R : float
            Radius of the circle.
        z_center : float
            z-coordinate of the circle center.
        z_cut : float
            z-coordinate of the cut line.

        Returns
        -------
        float
            Area of the circular segment.
        """
        h = (z_center + R) - z_cut  # height of the cap
        if h <= 0:
            return 0.0
        if h >= 2 * R:
            return np.pi * R**2
        if h <= R:
            return R**2 * np.arccos((R - h) / R) - (R - h) * np.sqrt(2 * R * h - h**2)
        else:
            h_small = 2 * R - h
            return np.pi * R**2 - (R**2 * np.arccos((R - h_small) / R) - (R - h_small) * np.sqrt(2 * R * h_small - h_small**2))

    def read_data(self):
        """
        Read and parse data from log files in each directory.
        """
        for directory in self.directories:
            log_path = os.path.join(directory, "log_data.txt")
            if os.path.isfile(log_path):
                with open(log_path, "r") as f:
                    text = f.read()
                R_eq = float(re.search(r"R_eq:([0-9\.\-eE]+)", text).group(1))
                zi_c = float(re.search(r"zi_c:([0-9\.\-eE]+)", text).group(1))
                zi_0 = float(re.search(r"zi_0:([0-9\.\-eE]+)", text).group(1))
                angle = float(re.search(r"Contact angle:([0-9\.\-eE]+)", text).group(1))
                A_seg = self.circular_segment_area(R_eq, zi_c, zi_0)
                self.data[directory]["R_eq"].append(R_eq)
                self.data[directory]["zi_c"].append(zi_c)
                self.data[directory]["zi_0"].append(zi_0)
                self.data[directory]["contact_angles"].append(angle)
                self.data[directory]["surface_areas"].append(A_seg)

    def analyze(self):
        """
        Analyze and print statistics for each directory.
        """
        self.read_data()
        for directory in self.directories:
            print(f"Directory: {directory}")
            print(f"  Mean Surface Area: {np.mean(self.data[directory]['surface_areas'])}")
            print(f"  Mean Contact Angle: {np.mean(self.data[directory]['contact_angles'])}")

    def plot_mean_angle_vs_surface(
        self, labels=None, colors=None, markers=None, save_path=None
    ):
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
            If provided, saves the figure to this path (e.g., "angle_vs_surface.png").
        """
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
            labels = [d.replace("_reduce_binned", "") for d in self.directories]
        if colors is None:
            colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(self.directories)))
        if markers is None:
            markers = ["o", "s", "^", "D", "v", "p", "h", "X"][:len(self.directories)]

        xvals, yvals = [], []
        for d, label, color, marker in zip(self.directories, labels, colors, markers):
            surface_areas = self.data[d]["surface_areas"]
            contact_angles = self.data[d]["contact_angles"]
            x = 1 / np.sqrt(np.mean(surface_areas))
            y = np.mean(contact_angles)
            yerr = np.std(contact_angles) / np.sqrt(len(contact_angles))  # SEM estimate
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

        # Linear fit
        xvals, yvals = np.array(xvals), np.array(yvals)
        coeffs = np.polyfit(xvals, yvals, 1)
        fit_line = np.poly1d(coeffs)
        x_fit = np.linspace(0, max(xvals) * 1.1, 100)
        ax.plot(x_fit, fit_line(x_fit), "--", color="gray", lw=1.5, label=f"Linear Fit (y = {fit_line(0):.2f}°)")

        ax.set_xlabel(r"$1 / \sqrt{\text{Surface Area}}$")
        ax.set_ylabel("Mean Angle (°)")
        ax.set_title("Mean Angle vs Surface Area Scaling", pad=10)
        ax.legend(frameon=False, loc="upper left")
        ax.grid(False)
        ax.set_xlim(left=-0.001)
        ax.set_ylim(bottom=min(yvals) - 2, top=max(yvals) + 2)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=400, bbox_inches="tight")
        plt.close()


# Example usage:
# analyzer = Binning_Trajectory_Analyzer(directories=["./result_dump_1", "./result_dump_2"])
# analyzer.analyze()
# analyzer.plot_mean_angle_vs_surface(save_path="mean_angle_vs_surface.png")