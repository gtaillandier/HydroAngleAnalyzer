import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class MethodComparison:
    """
    Compare different trajectory analysis methods.
    """
    
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
        markers = ["o", "s", "^", "D", "v", "p", "h", "X"]
        
        for idx, (analyzer, ax, method_name, colors) in enumerate(
            zip(self.analyzers, axes, self.method_names, colors_list)
        ):
            xvals, yvals = [], []
            
            for i, directory in enumerate(analyzer.directories):
                x, y, yerr = analyzer.compute_statistics(directory)
                label = analyzer.get_clean_label(directory)
                marker = markers[i % len(markers)]
                
                ax.errorbar(
                    x, y, yerr=yerr, fmt=marker, color=colors[i],
                    markersize=6, capsize=3, lw=1.2
                )
                ax.annotate(
                    label, xy=(x, y), xytext=(5, 5), textcoords="offset points",
                    ha="left", va="center", fontsize=6, color="black"
                )
                xvals.append(x)
                yvals.append(y)
            
            # Linear fit
            xvals_arr, yvals_arr = np.array(xvals), np.array(yvals)
            coeffs = np.polyfit(xvals_arr, yvals_arr, 1)
            fit_line = np.poly1d(coeffs)
            x_fit = np.linspace(0, max(xvals_arr) * 1.1, 100)
            ax.plot(
                x_fit, fit_line(x_fit), "--", color="gray", lw=1.5,
                label=f"Fit: y={coeffs[0]:.2f}x+{fit_line(0):.2f}°"
            )
            
            ax.set_xlabel(r"$1 / \sqrt{\text{Surface Area}}$")
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
        
        # Different color schemes for different methods
        method_colors = [plt.cm.Set1(i) for i in range(len(self.analyzers))]
        markers = ["o", "s", "^", "D", "v", "p", "h", "X"]
        
        all_xvals, all_yvals = [], []
        
        for method_idx, (analyzer, method_name, base_color) in enumerate(
            zip(self.analyzers, self.method_names, method_colors)
        ):
            xvals, yvals = [], []
            
            # Create lighter shades of the base color for each directory
            n_dirs = len(analyzer.directories)
            color_variations = [
                tuple(list(base_color[:3]) + [0.4 + 0.6 * i / max(n_dirs - 1, 1)])
                for i in range(n_dirs)
            ]
            
            for i, directory in enumerate(analyzer.directories):
                x, y, yerr = analyzer.compute_statistics(directory)
                label = f"{method_name}: {analyzer.get_clean_label(directory)}"
                marker = markers[(method_idx * 2 + i) % len(markers)]
                
                ax.errorbar(
                    x, y, yerr=yerr, fmt=marker, color=base_color,
                    markersize=6, capsize=3, lw=1.2, alpha=0.7,
                    label=label
                )
                xvals.append(x)
                yvals.append(y)
            
            all_xvals.extend(xvals)
            all_yvals.extend(yvals)
            
            # Linear fit for this method
            xvals_arr, yvals_arr = np.array(xvals), np.array(yvals)
            coeffs = np.polyfit(xvals_arr, yvals_arr, 1)
            fit_line = np.poly1d(coeffs)
            x_fit = np.linspace(0, max(xvals_arr) * 1.1, 100)
            ax.plot(
                x_fit, fit_line(x_fit), "--", color=base_color, lw=2,
                label=f"{method_name} Fit: {fit_line(0):.2f}°"
            )
        
        ax.set_xlabel(r"$1 / \sqrt{\text{Surface Area}}$")
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
                angles = analyzer.get_contact_angles(directory)
                surfaces = analyzer.get_surface_areas(directory)
                
                all_angles.extend(angles)
                all_surfaces.extend(surfaces)
                
                print(f"  {analyzer.get_clean_label(directory)}:")
                print(f"    Mean Surface Area: {np.mean(surfaces):.4f} ± {np.std(surfaces):.4f}")
                print(f"    Mean Angle: {np.mean(angles):.4f}° ± {np.std(angles):.4f}°")
            
            print(f"\n  Overall Statistics:")
            print(f"    Total samples: {len(all_angles)}")
            print(f"    Mean Surface Area: {np.mean(all_surfaces):.4f}")
            print(f"    Mean Angle: {np.mean(all_angles):.4f}°")
            print(f"    Std Angle: {np.std(all_angles):.4f}°")
        
        print("\n" + "=" * 70)


# Example usage:
if __name__ == "__main__":
    from sliced_trajectory_analyzer import SlicedTrajectoryAnalyzer
    from binning_trajectory_analyzer import BinningTrajectoryAnalyzer
    
    # Define directories for each method
    sliced_dirs = ["./result_dump_1_reduce_sliced", "./result_dump_2_reduce_sliced"]
    binned_dirs = ["./result_dump_1_reduce_binned", "./result_dump_2_reduce_binned"]
    
    # Create analyzers
    sliced_analyzer = SlicedTrajectoryAnalyzer(directories=sliced_dirs)
    binned_analyzer = BinningTrajectoryAnalyzer(directories=binned_dirs)
    
    # Create comparison
    comparison = MethodComparison(
        analyzers=[sliced_analyzer, binned_analyzer],
        method_names=["Sliced Method", "Binning Method"]
    )
    
    # Generate comparison plots
    comparison.plot_side_by_side_comparison(save_path="method_comparison_side_by_side.png")
    comparison.plot_overlay_comparison(save_path="method_comparison_overlay.png")
    
    # Print statistics
    comparison.compare_statistics()