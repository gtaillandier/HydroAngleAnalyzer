import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List


class AngleAnalyzerPlotter:
    """
    Class to visualize results from hydroangleanalyzer contact angle analysis.
    It can plot surface profiles, fitted circles, tangent lines, and distributions of contact angles.
    """

    def __init__(self, results_dir: str):
        """
        Initialize the plotter with the directory containing the results.

        Parameters
        ----------
        results_dir : str
            Path to the directory with saved .npy and .txt analysis files.
        """
        self.results_dir = results_dir
        self.surfaces = {}
        self.contact_angles = None

    # -------------------------------------------------
    # Data Loading Methods
    # -------------------------------------------------
    def load_surfaces(self, frame_ids: Optional[List[int]] = None):
        """Load surface points (.npy) from the results directory."""
        files = [f for f in os.listdir(self.results_dir) if f.startswith("surfacesframe") and f.endswith(".npy")]
        for f in files:
            frame_num = int(f.replace("surfacesframe", "").replace(".npy", ""))
            if frame_ids is None or frame_num in frame_ids:
                self.surfaces[frame_num] = np.load(os.path.join(self.results_dir, f), allow_pickle=True)

    def load_angles(self):
        """Load per-frame mean contact angles from combined results file."""
        file_path = os.path.join(self.results_dir, "alfas_per_frame_combined.txt")
        if os.path.exists(file_path):
            self.contact_angles = np.loadtxt(file_path)
        else:
            print("[WARNING] No 'alfas_per_frame_combined.txt' found in", self.results_dir)

    # -------------------------------------------------
    # Plotting Methods
    # -------------------------------------------------
    def plot_surface_profiles(self, frame_id: int, save: bool = False):
        """Plot the reconstructed droplet surfaces for a given frame."""
        if frame_id not in self.surfaces:
            raise ValueError(f"No surface data found for frame {frame_id}. Load with load_surfaces().")

        points = self.surfaces[frame_id]
        plt.figure(figsize=(8, 6))

        for i, sublist in enumerate(points):
            x_coords = sublist[:, 0]
            y_coords = sublist[:, 1]
            plt.plot(x_coords, y_coords, marker='o', linestyle='-', label=f'Curve {i + 1}')

        plt.xlabel('X [Å]')
        plt.ylabel('Y [Å]')
        plt.title(f'Surface Profile – Frame {frame_id}')
        plt.legend()
        plt.grid(True)

        if save:
            save_path = os.path.join(self.results_dir, f'surface_frame_{frame_id}.png')
            plt.savefig(save_path, dpi=300)
            print(f"[INFO] Saved plot to {save_path}")

        plt.show()

    def plot_contact_angle_distribution(self, save: bool = False):
        """Plot histogram and trend of mean contact angles per frame."""
        if self.contact_angles is None:
            raise ValueError("Contact angles not loaded. Use load_angles().")

        frames = self.contact_angles[:, 0]
        angles = self.contact_angles[:, 1]

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # Plot histogram
        ax[0].hist(angles, bins=15, color='skyblue', edgecolor='black')
        ax[0].set_xlabel('Contact Angle [°]')
        ax[0].set_ylabel('Frequency')
        ax[0].set_title('Distribution of Contact Angles')

        # Plot trend vs frame
        ax[1].plot(frames, angles, marker='o', linestyle='-', color='black')
        ax[1].set_xlabel('Frame')
        ax[1].set_ylabel('Mean Contact Angle [°]')
        ax[1].set_title('Contact Angle Evolution')

        plt.tight_layout()

        if save:
            save_path = os.path.join(self.results_dir, "contact_angle_statistics.png")
            plt.savefig(save_path, dpi=300)
            print(f"[INFO] Saved plot to {save_path}")

        plt.show()