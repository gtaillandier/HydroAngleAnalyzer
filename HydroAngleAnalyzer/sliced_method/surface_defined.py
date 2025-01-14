import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

class SurfaceDefinition:
    def __init__(self, atom_coords, center_geom, density_conversion=1.0):
        self.atom_coords = atom_coords
        self.center_geom = center_geom
        self.density_conversion = density_conversion

    @staticmethod
    def density_contribution(positions, coords, sigma=2.0):
        """
        Calculate the density contribution of atoms at given coordinates.

        Parameters:
        positions : array-like
            Positions to evaluate density.
        coords : array-like
            Atom coordinates.
        sigma : float, optional
            Standard deviation for Gaussian distribution.

        Returns:
        array-like
            Density contributions at given positions.
        """
        sigma2 = sigma * sigma
        prefactor = 1.0 / (2 * np.pi * sigma2)**1.5
        differences = positions[:, np.newaxis, :] - coords[np.newaxis, :, :]
        ri2 = np.sum(differences**2, axis=-1)
        den_contributions = prefactor * np.exp(-ri2 / (2 * sigma2))
        return np.sum(den_contributions, axis=1)

    @staticmethod
    def density_profile(z, zd, d, h):
        """
        Function to model the density profile.

        Parameters:
        z : array-like
            Positions.
        zd : float
            Parameter for fitting.
        d : float
            Parameter for fitting.
        h : float
            Parameter for fitting.

        Returns:
        array-like
            Density profile.
        """
        return (np.tanh(-z + zd) * d + h)

    def fit_density_profile(self, z_data, density, param_bounds):
        """
        Fit the function to the density data to find parameters.

        Parameters:
        z_data : array-like
            Positions.
        density : array-like
            Density values.
        param_bounds : tuple
            Bounds for parameters.

        Returns:
        float
            Fitted parameter re.
        """
        popt, _ = curve_fit(self.density_profile, z_data, density, bounds=param_bounds)
        zd, d, h = popt
        return zd

    def analyze_lines(self, delta_angle, nn, max_dist, gamma):
        """
        Calculate the density profile along multiple lines.

        Parameters:
        delta_angle : float
            Angle increment for lines.
        nn : int
            Number of points per line.
        max_dist : float
            Maximum distance to consider.
        gamma : float
            Angle parameter.

        Returns:
        tuple
            Lists of interface positions and XZ coordinates.
        """
        beta = np.linspace(0, 360, int(360 / delta_angle), endpoint=False)
        list_rr = []
        list_xz = []
        param_bounds = ([0, -10, -10], [max_dist, 10, 10])

        cos_beta = np.cos(np.deg2rad(beta))
        sin_beta = np.sin(np.deg2rad(beta))
        cos_gamma = np.cos(np.deg2rad(gamma))
        sin_gamma = np.sin(np.deg2rad(gamma))

        for i in range(len(beta)):
            x_dir = cos_beta[i] * cos_gamma
            y_dir = sin_gamma * cos_beta[i]
            z_dir = sin_beta[i]

            direction = np.array([x_dir, y_dir, z_dir])
            positions = np.linspace(self.center_geom, self.center_geom + max_dist * direction, nn)
            distances = np.linspace(0.0, max_dist, nn)
            sigma = 3
            density = self.density_conversion * self.density_contribution(positions, self.atom_coords, sigma=sigma)
            interface_re = self.fit_density_profile(distances, density, param_bounds)
            list_rr.append([interface_re, beta[i]])
            list_xz.append([cos_beta[i] * interface_re + self.center_geom[0],
                            sin_beta[i] * interface_re + self.center_geom[2]])

        return list_rr, list_xz

# Example usage:
# surface_def = SurfaceDefinition(atom_coords, center_geom)
# list_rr, list_xz = surface_def.analyze_lines(delta_angle=10, nn=100, max_dist=50, gamma=30)