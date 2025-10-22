import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

plt.style.use('seaborn-v0_8-whitegrid')


class Droplet_sliced_Plotter:
    """
    Class for plotting droplet surfaces and fitted contact angles.

    Parameters
    ----------
    center : bool, optional
        Whether to recenter z-coordinates around the mean wall height. Default is True.
    show_wall : bool, optional
        Whether to plot wall particles. Default is True.
    molecule_view : bool, optional
        Whether to plot water as molecules (O + 2 H atoms). If False, plots only oxygen points. Default is True.
    """

    def __init__(self, center: bool = True, show_wall: bool = True, molecule_view: bool = True):
        self.center = center
        self.show_wall = show_wall
        self.molecule_view = molecule_view

        # Colors
        self.oxygen_color = '#d62828'
        self.hydrogen_color = 'white'
        self.surface_color = 'black'
        self.circle_color = '#0A9396'
        self.wall_color = 'black'

    def plot_surface_points(self, oxygen_position, surface_data, popt, wall_coords,
                            output_filename, y_com=None, pbc_y=None):
        """
        Plot the droplet surface, fitted circle, and optionally wall atoms and water molecules.
        """

        if y_com is None:
            y_com = np.mean(oxygen_position[:, 1])

        # Select atoms near the Y center (±3 Å)
        if pbc_y is not None:
            dy = np.abs(oxygen_position[:, 1] - y_com)
            dy = np.minimum(dy, pbc_y - dy)
            mask = dy <= 3
        else:
            mask = np.abs(oxygen_position[:, 1] - y_com) <= 3
        oxygen_selected = oxygen_position[mask]

        # --- Subsample for clarity ---
        rng = np.random.default_rng(42)
        keep_fraction = 0.70
        sample_idx = rng.choice(len(oxygen_selected),
                                size=int(len(oxygen_selected) * keep_fraction),
                                replace=False)
        oxygen_selected = oxygen_selected[sample_idx]

        # --- Limit wall region under droplet (±5 Å margin) ---
        x_min, x_max = np.min(oxygen_selected[:, 0]) - 5, np.max(oxygen_selected[:, 0]) + 5
        wall_mask = (wall_coords[:, 0] >= x_min) & (wall_coords[:, 0] <= x_max)
        wall_coords = wall_coords[wall_mask]

        # --- Optional recentring ---
        if self.center:
            z_shift = np.mean(wall_coords[:, 2])
            oxygen_selected[:, 2] -= z_shift
            wall_coords[:, 2] -= z_shift
            surface_data = [np.column_stack([surf[:, 0], surf[:, 1] - z_shift]) for surf in surface_data]
            Xc, Zc, R, limit_med = popt
            Zc -= z_shift
        else:
            Xc, Zc, R, limit_med = popt

        # --- Plot setup ---
        fig, ax = plt.subplots(figsize=(4.0, 3.0), dpi=300)

        # --- Wall atoms ---
        if self.show_wall:
            ax.scatter(wall_coords[:, 0], wall_coords[:, 2], color=self.wall_color,
                       s=3, alpha=0.7, zorder=0)

        # --- Water representation ---
        if self.molecule_view:
            h_dist = 1.0
            for ox, oy, oz in oxygen_selected:
                ax.scatter(ox, oz, color=self.oxygen_color, s=8, alpha=0.9,
                           edgecolors='none', linewidths=0.15, zorder=1)
                for _ in range(2):
                    angle = rng.uniform(0, 2 * np.pi)
                    dx, dz = h_dist * np.cos(angle), h_dist * np.sin(angle)
                    ax.scatter(ox + dx, oz + dz, color=self.hydrogen_color, s=4, alpha=0.8,
                               edgecolors='black', linewidths=0.15, zorder=1)
        else:
            ax.scatter(oxygen_selected[:, 0], oxygen_selected[:, 2],
                       color=self.oxygen_color, s=6, alpha=0.9, zorder=1)

        # --- Surface line ---
        for surf in surface_data:
            X_data, Z_data = surf[:, 0], surf[:, 1]
            if not np.allclose([X_data[0], Z_data[0]], [X_data[-1], Z_data[-1]]):
                X_data = np.append(X_data, X_data[0])
                Z_data = np.append(Z_data, Z_data[0])
            ax.plot(X_data, Z_data, color=self.surface_color, lw=1.5, zorder=3)

        # --- Fitted circle ---
        circle = plt.Circle((Xc, Zc), R, color=self.circle_color,
                            fill=False, ls='--', lw=2.5, zorder=4)
        ax.add_artist(circle)

        # --- Axes ---
        ax.set_xlabel("x (Å)", fontsize=9)
        ax.set_ylabel("z (Å)", fontsize=9)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_aspect('equal', adjustable='box')
        ax.grid(False)
        ax.set_xlim(x_min - 5, x_max + 5)

        # --- Legend ---
        ax.legend(
            handles=[
                plt.Line2D([], [], color=self.surface_color, lw=1.5, label='Surface contour'),
                plt.Line2D([], [], color=self.circle_color, ls='--', lw=1.5, label='Fitted circle')
            ],
            loc='upper left',
            frameon=False,
            fontsize=7
        )

        plt.tight_layout(pad=0.1)
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
