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
    def plot_combined(
        self,
        frame_id: int,
        include_liquid: bool = False,
        positions: Optional[np.ndarray] = None,
        save: bool = True,
        show: bool = True
    ):
        """
        Plot both the fitted circle(s) and the surface profile for a given frame.
        Optionally include liquid particle positions (either loaded from file or provided directly).

        Parameters
        ----------
        frame_id : int
            Frame number to visualize.
        include_liquid : bool, optional
            If True, overlay liquid particle positions (from 'liquidframe{frame_id}.npy' or 'positions' argument).
        positions : np.ndarray, optional
            Directly provide positions as an array of shape (N, 2) or (N, 3).
            If given, it overrides loading from file.
        save : bool
            Whether to save the figure as PNG and SVG.
        show : bool
            Whether to display the plot interactively.
        """
        # Load required data
        surfaces = self.load_surface(frame_id)
        popts = self.load_popt(frame_id)
        mean_angle = self.load_angle(frame_id)

        # Determine if we have liquid positions
        liquid_positions = None
        if include_liquid:
            if positions is not None:
                liquid_positions = positions
            else:
                liquid_path = os.path.join(self.results_dir, f"liquidframe{frame_id}.npy")
                if os.path.exists(liquid_path):
                    liquid_positions = np.load(liquid_path)
                else:
                    print(f"[WARNING] No liquid particle data available for frame {frame_id}.")

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 7))

        # (1) Plot surface points
        for i, segment in enumerate(surfaces):
            x_coords = segment[:, 0]
            z_coords = segment[:, 1]
            ax.plot(
                x_coords, z_coords,
                marker='o', linestyle='-', linewidth=1.3,
                label=f'Surface {i + 1}', alpha=0.9
            )

        # (2) Plot fitted circles
        for j, (xc, zc, r) in enumerate(popts):
            theta = np.linspace(0, 2 * np.pi, 300)
            x_circle = xc + r * np.cos(theta)
            z_circle = zc + r * np.sin(theta)
            ax.plot(x_circle, z_circle, '--', linewidth=1.4, label=f'Fitted Circle {j + 1}')
            ax.scatter(xc, zc, color='red', s=40, zorder=5, label=f'Center {j + 1}' if j == 0 else None)

        # (3) Optionally add liquid particles
        if liquid_positions is not None:
            if liquid_positions.shape[1] == 3:
                x, z = liquid_positions[:, 0], liquid_positions[:, 2]
            else:
                x, z = liquid_positions[:, 0], liquid_positions[:, 1]
            ax.scatter(x, z, s=10, c='blue', alpha=0.4, label='Liquid Particles')

        # (4) Aesthetics
        ax.set_xlabel('X (Å)', fontsize=14)
        ax.set_ylabel('Z (Å)', fontsize=14)
        ax.set_title(f"Frame {frame_id}: Surface, Fitted Circle(s), and Contact Angle = {mean_angle:.2f}°", fontsize=15)
        ax.legend(fontsize=10)
        ax.grid(True)
        ax.set_aspect('equal', adjustable='datalim')

        # (5) Save figure
        if save:
            suffix = f"frame_{frame_id}_combined"
            png_path = os.path.join(self.results_dir, f"{suffix}.png")
            svg_path = os.path.join(self.results_dir, f"{suffix}.svg")
            plt.savefig(png_path, dpi=300)
            plt.savefig(svg_path)
            print(f"[INFO] Saved combined plot to:\n  - {png_path}\n  - {svg_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)
        
    def plot_liquid_particles(self, frame_id: int = None, positions: np.ndarray = None, save: bool = True, show: bool = True):
        """
        Plot the liquid particle positions for a specific frame or from a given array.

        Parameters
        ----------
        frame_id : int, optional
            Frame number to load positions from 'liquidframe{frame_id}.npy'.
        positions : np.ndarray, optional
            Directly provide positions as a (N, 2) or (N, 3) array.
        save : bool
            Whether to save the figure as PNG and SVG.
        show : bool
            Whether to display the plot interactively.
        """
        if positions is None:
            if frame_id is None:
                raise ValueError("Either frame_id or positions must be provided.")
            path = os.path.join(self.results_dir, f"liquidframe{frame_id}.npy")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Liquid particle file not found: {path}")
            positions = np.load(path)
        # Use only X and Z if 3D
        if positions.shape[1] == 3:
            x, z = positions[:, 0], positions[:, 2]
        else:
            x, z = positions[:, 0], positions[:, 1]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(x, z, s=10, c='blue', alpha=0.6, label='Liquid Particles')
        ax.set_xlabel('X (Å)', fontsize=14)
        ax.set_ylabel('Z (Å)', fontsize=14)
        title = f'Liquid Particle Positions'
        if frame_id is not None:
            title += f' – Frame {frame_id}'
        ax.set_title(title, fontsize=15)
        ax.legend(fontsize=10)
        ax.grid(True)
        ax.set_aspect('equal', adjustable='datalim')
        if save:
            suffix = f"frame_{frame_id}_liquid_particles" if frame_id is not None else "liquid_particles"
            png_path = os.path.join(self.results_dir, f"{suffix}.png")
            svg_path = os.path.join(self.results_dir, f"{suffix}.svg")
            plt.savefig(png_path, dpi=300)
            plt.savefig(svg_path)
            print(f"[INFO] Saved liquid particle plots to:\n  - {png_path}\n  - {svg_path}")
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