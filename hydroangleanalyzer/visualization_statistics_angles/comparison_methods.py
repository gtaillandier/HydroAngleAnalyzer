import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class MethodComparison:
    def __init__(self, analyzers, method_names=None):
        """
        Initialize comparison with multiple analyzers.

        Parameters
        ----------
        analyzers : list of BaseTrajectoryAnalyzer
            List of analyzer instances to compare.
        method_names : list of str, optional
            Custom names for each method. If None, uses get_method_name().
        """
        self.analyzers = analyzers
        self.method_names = method_names or [a.get_method_name() for a in analyzers]
        # Ensure all analyzers have read data
        for analyzer in self.analyzers:
            if not hasattr(analyzer, 'data') or not analyzer.data:
                analyzer.read_data()

    def _check_and_run_analysis(self, analyzer):
        """Check for analysis output files, run analysis if missing."""
        for directory in analyzer.directories:
            output_file = f"{directory}/output_stats.txt"
            if not os.path.exists(output_file):
                print(f"No analysis found for {directory}. Running analysis...")
                analyzer.analyze()
                break

    def _read_analysis_output(self, analyzer, directory):
        """Read mean surface area and mean angle from analysis output file."""
        output_file = f"{directory}/output_stats.txt"
        with open(output_file, 'r') as f:
            lines = f.readlines()
            mean_surface_area = float(lines[2].split(": ")[1].strip())
            mean_contact_angle = float(lines[3].split(": ")[1].strip())
        return mean_surface_area, mean_contact_angle

    def plot_side_by_side_comparison(self, save_path=None, figsize=(14, 5)):
        """
        Create side-by-side plots comparing both methods.

        Parameters
        ----------
        save_path : str, optional
            Path to save the figure.
        figsize : tuple, optional
            Figure size (width, height).
        """
        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 13,
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "legend.fontsize": 11,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "axes.linewidth": 1.0,
            "errorbar.capsize": 3,
        })

        fig, axes = plt.subplots(1, len(self.analyzers), figsize=figsize)
        if len(self.analyzers) == 1:
            axes = [axes]

        colors_list = [
            plt.cm.viridis(np.linspace(0.15, 0.85, len(analyzer.directories)))
            for analyzer in self.analyzers
        ]

        for idx, (analyzer, ax, method_name, colors) in enumerate(
            zip(self.analyzers, axes, self.method_names, colors_list)
        ):
            self._check_and_run_analysis(analyzer)
            xvals, yvals = [], []
            for i, directory in enumerate(analyzer.directories):
                try:
                    mean_surface_area, mean_contact_angle = self._read_analysis_output(analyzer, directory)
                except FileNotFoundError:
                    x, y, yerr = analyzer.compute_statistics(directory)
                    mean_surface_area, mean_contact_angle = np.mean(x), np.mean(y)
                x_time = np.arange(len(x)) * analyzer.time_step if hasattr(analyzer, 'time_step') else np.array([0])
                label = analyzer.get_clean_label(directory)
                ax.errorbar(
                    x_time, mean_contact_angle, yerr=0.5, color=colors[i],
                    markersize=6, capsize=3, lw=1.2
                )
                xvals.append(x_time)
                yvals.append(mean_contact_angle)

            xvals_arr = np.concatenate(xvals)
            yvals_arr = np.concatenate(yvals)
            coeffs = np.polyfit(xvals_arr, yvals_arr, 1)
            fit_line = np.poly1d(coeffs)
            x_fit = np.linspace(0, max(xvals_arr) * 1.1, 100)
            ax.plot(
                x_fit, fit_line(x_fit), "--", color="gray", lw=1.5,
                label=f"Fit: y={coeffs[0]:.2f}x+{fit_line(0):.2f}°"
            )
            ax.set_xlabel(f"Time ({getattr(analyzer, 'time_unit', 'fs')})")
            ax.set_ylabel("Mean Angle (°)")
            ax.set_title(f"{method_name}", pad=10)
            ax.legend(frameon=False, loc="upper left", fontsize=9)
            ax.grid(False)
            ax.set_xlim(left=-0.001)
            ax.set_ylim(bottom=min(yvals) - 2, top=max(yvals) + 2)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=400, bbox_inches="tight")
        plt.show()

    def plot_overlay_comparison(self, save_path=None, figsize=(8, 6)):
        """
        Create an overlay plot comparing both methods on the same axes.

        Parameters
        ----------
        save_path : str, optional
            Path to save the figure.
        figsize : tuple, optional
            Figure size (width, height).
        """
        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 13,
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "legend.fontsize": 10,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "axes.linewidth": 1.0,
            "errorbar.capsize": 3,
        })

        fig, ax = plt.subplots(figsize=figsize)
        method_colors = [plt.cm.Set1(i) for i in range(len(self.analyzers))]
        all_xvals, all_yvals = [], []

        for method_idx, (analyzer, method_name, base_color) in enumerate(
            zip(self.analyzers, self.method_names, method_colors)
        ):
            self._check_and_run_analysis(analyzer)
            xvals, yvals = [], []
            for i, directory in enumerate(analyzer.directories):
                try:
                    mean_surface_area, mean_contact_angle = self._read_analysis_output(analyzer, directory)
                except FileNotFoundError:
                    x, y, yerr = analyzer.compute_statistics(directory)
                    mean_surface_area, mean_contact_angle = np.mean(x), np.mean(y)
                x_time = np.arange(len(x)) * analyzer.time_step if hasattr(analyzer, 'time_step') else np.array([0])
                label = f"{method_name}: {analyzer.get_clean_label(directory)}"
                ax.errorbar(
                    x_time, mean_contact_angle, yerr=0.5, color=base_color,
                    markersize=6, capsize=3, lw=1.2, alpha=0.7,
                    label=label
                )
                xvals.append(x_time)
                yvals.append(mean_contact_angle)

            xvals_arr = np.concatenate(xvals)
            yvals_arr = np.concatenate(yvals)
            coeffs = np.polyfit(xvals_arr, yvals_arr, 1)
            fit_line = np.poly1d(coeffs)
            x_fit = np.linspace(0, max(xvals_arr) * 1.1, 100)
            ax.plot(
                x_fit, fit_line(x_fit), "--", color=base_color, lw=2,
                label=f"{method_name} Fit: {fit_line(0):.2f}°"
            )
            all_xvals.extend(xvals)
            all_yvals.extend(yvals)

        ax.set_xlabel(f"Time ({getattr(self.analyzers[0], 'time_unit', 'fs')})")
        ax.set_ylabel("Mean Angle (°)")
        ax.set_title("Method Comparison: Mean Angle vs Surface Area", pad=10)
        ax.legend(frameon=False, loc="upper left", fontsize=8)
        ax.grid(False)
        ax.set_xlim(left=-0.001)
        if all_yvals:
            ax.set_ylim(bottom=min(all_yvals) - 2, top=max(all_yvals) + 2)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=400, bbox_inches="tight")
        plt.show()

    def compare_statistics(self):
        """
        Print a statistical comparison of both methods.
        """
        print("=" * 70)
        print("METHOD COMPARISON STATISTICS")
        print("=" * 70)

        for method_name, analyzer in zip(self.method_names, self.analyzers):
            print(f"\n{method_name}:")
            print("-" * 70)

            all_angles = []
            all_surfaces = []

            for directory in analyzer.directories:
                try:
                    mean_surface_area, mean_contact_angle = self._read_analysis_output(analyzer, directory)
                except FileNotFoundError:
                    angles = analyzer.get_contact_angles(directory)
                    surfaces = analyzer.get_surface_areas(directory)
                    mean_surface_area, mean_contact_angle = np.mean(surfaces), np.mean(angles)

                all_angles.extend(angles) if 'angles' in locals() else all_angles.extend(analyzer.get_contact_angles(directory))
                all_surfaces.extend(surfaces) if 'surfaces' in locals() else all_surfaces.extend(analyzer.get_surface_areas(directory))

                print(f"  {analyzer.get_clean_label(directory)}:")
                print(f"    Mean Surface Area: {mean_surface_area:.4f}")
                print(f"    Mean Angle: {mean_contact_angle:.4f}°")

            print(f"\n  Overall Statistics:")
            print(f"    Total samples: {len(all_angles)}")
            print(f"    Mean Surface Area: {np.mean(all_surfaces):.4f}")
            print(f"    Mean Angle: {np.mean(all_angles):.4f}°")
            print(f"    Std Angle: {np.std(all_angles):.4f}°")

        print("\n" + "=" * 70)
