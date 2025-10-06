# hydroangleanalyzer/angles_frames_analysis/graphs_circle_surfaces.py

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple


class GraphsCircleSurfaces:
    """
    A visualization utility for plotting droplet surface profiles,
    fitted circles, and contact angles from HydroAngleAnalyzer results.
    """

    def __init__(self, results_dir: str):
        """
        Initialize with the directory containing post-processed simulation results.

        Parameters
        ----------
        results_dir : str
            Directory containing .npy and .txt files (surfacesframe*, poptsframe*, alfasframe*).
        """
        self.results_dir = results_dir
        self.surfaces = {}
        self.popts = {}
        self.angles = {}

    # -------------------------------------------------
    # Data Loading
    # -------------------------------------------------
    def load_surface(self, frame_id: int):
        """Load a specific frame's surface points."""
        path = os.path.join(self.results_dir, f"surfacesframe{frame_id}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Surface file not found: {path}")
        self.surfaces[frame_id] = np.load(path, allow_pickle=True)
        return self.surfaces[frame_id]

    def load_popt(self, frame_id: int):
        """Load fitted circle parameters (x_center, z_center, radius)."""
        path = os.path.join(self.results_dir, f"poptsframe{frame_id}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fitted circle file not found: {path}")
        self.popts[frame_id] = np.load(path)
        return self.popts[frame_id]

    def load_angle(self, frame_id: int):
        """Load mean contact angle for a frame."""
        path = os.path.join(self.results_dir, f"alfasframe{frame_id}.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Angle file not found: {path}")
        value = np.loadtxt(path)
        self.angles[frame_id] = float(np.mean(value))
        return self.angles[frame_id]

    # -------------------------------------------------
    # Plotting
    # -------------------------------------------------
    def plot_frame(self, frame_id: int, save: bool = True, show: bool = True):
        """
        Plot the surface points and fitted circles for a specific frame.

        Parameters
        ----------
        frame_id : int
            Frame number to visualize.
        save : bool
            Whether to save the figure as PNG and SVG.
        show : bool
            Whether to display the plot interactively.
        """
        # Load data
        surfaces = self.load_surface(frame_id)
        popts = self.load_popt(frame_id)
        mean_angle = self.load_angle(frame_id)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 7))

        # Plot each surface segment
        for i, segment in enumerate(surfaces):
            x_coords = segment[:, 0]
            z_coords = segment[:, 1]
            ax.plot(x_coords, z_coords, marker='o', linestyle='-', label=f'Surface {i + 1}')

        # Plot fitted circles
        for j, (xc, zc, r) in enumerate(popts):
            theta = np.linspace(0, 2 * np.pi, 200)
            x_circle = xc + r * np.cos(theta)
            z_circle = zc + r * np.sin(theta)
            ax.plot(x_circle, z_circle, '--', label=f'Fitted Circle {j + 1}')

        # Aesthetic options
        ax.set_xlabel('X (Å)', fontsize=14)
        ax.set_ylabel('Z (Å)', fontsize=14)
        ax.set_title(f'Droplet Surface and Fitted Circles – Frame {frame_id}\nMean Contact Angle: {mean_angle:.2f}°', fontsize=15)
        ax.legend(fontsize=10)
        ax.grid(True)
        ax.set_aspect('equal', adjustable='datalim')

        # Save if required
        if save:
            png_path = os.path.join(self.results_dir, f"frame_{frame_id}_circles.png")
            svg_path = os.path.join(self.results_dir, f"frame_{frame_id}_circles.svg")
            plt.savefig(png_path, dpi=300)
            plt.savefig(svg_path)
            print(f"[INFO] Saved plots to:\n  - {png_path}\n  - {svg_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    # -------------------------------------------------
    # Multi-frame summary
    # -------------------------------------------------
    def plot_angle_evolution(self, frame_ids: Optional[List[int]] = None, save: bool = True):
        """Plot evolution of mean contact angle vs frame number."""
        if not self.angles:
            # Load all automatically if not already loaded
            for f in self._detect_all_frames():
                self.load_angle(f)

        frames = sorted(frame_ids if frame_ids else list(self.angles.keys()))
        values = [self.angles[f] for f in frames]

        plt.figure(figsize=(8, 5))
        plt.plot(frames, values, marker='o', color='black', linestyle='-')
        plt.xlabel("Frame", fontsize=13)
        plt.ylabel("Mean Contact Angle (°)", fontsize=13)
        plt.title("Evolution of Contact Angle", fontsize=14)
        plt.grid(True)

        if save:
            save_path = os.path.join(self.results_dir, "contact_angle_evolution.png")
            plt.savefig(save_path, dpi=300)
            print(f"[INFO] Saved contact angle evolution plot to {save_path}")

        plt.show()

    # -------------------------------------------------
    # Utility
    # -------------------------------------------------
    def _detect_all_frames(self) -> List[int]:
        """Automatically detect available frames in the results directory."""
        frames = []
        for f in os.listdir(self.results_dir):
            if f.startswith("surfacesframe") and f.endswith(".npy"):
                frame = int(f.replace("surfacesframe", "").replace(".npy", ""))
                frames.append(frame)
        return sorted(frames)