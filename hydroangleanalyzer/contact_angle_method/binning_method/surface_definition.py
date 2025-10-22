import numpy as np
from scipy.optimize import curve_fit

class SurfaceModel:
    """Base class for surface models used in contact angle analysis."""

    def __init__(self, initial_params=None):
        """Initialize the surface model with optional initial parameters."""
        self.params = initial_params
        self.covariance = None

    def fit(self, x_data, density_data):
        """Fit the model to density data."""
        raise NotImplementedError("Subclasses must implement this method")

    def evaluate(self, x):
        """Evaluate the fitted function at point x."""
        raise NotImplementedError("Subclasses must implement this method")

    def evaluate_on_grid(self, xi_grid, zi_grid):
        """Evaluate the fitted function on a 2D grid."""
        out_fitted = np.zeros((len(xi_grid), len(zi_grid)))
        for i in range(len(xi_grid)):
            for j in range(len(zi_grid)):
                out_fitted[i, j] = self.evaluate((xi_grid[i], zi_grid[j]))
        return out_fitted

class HyperbolicTangentModel(SurfaceModel):
    """Model based on hyperbolic tangent function for liquid-vapor interface."""

    def __init__(self, initial_params=None):
        """
        Initialize with default or provided parameters.
        
        Parameters:
        initial_params: list with 7 elements [rho1, rho2, R_eq, zi_c, zi_0, t1, t2]
            rho1: density of liquid phase
            rho2: density of vapor phase
            R_eq: radius of the sphere
            zi_c: z-coordinate of sphere center
            zi_0: z-coordinate reference
            t1: interface thickness parameter for radial component
            t2: interface thickness parameter for z component
        """
        if initial_params is None:
            initial_params = [1e-3, 3e-2, 40.0, 20.0, 4.0, 1.0, 1.0]
        super().__init__(initial_params)

    def _fitting_function(self, x, rho1, rho2, R_eq, zi_c, zi_0, t1, t2):
        """
        Define the fitting function based on a hyperbolic tangent model.
        
        Parameters:
        x: tuple (xi, zi) of coordinates
        rho1: density of liquid phase
        rho2: density of vapor phase
        R_eq: radius of the sphere
        zi_c: z-coordinate of sphere center
        zi_0: z-coordinate reference
        t1: interface thickness parameter for radial component
        t2: interface thickness parameter for z component
        """
        xi, zi = x[0], x[1]
        g = lambda r: 0.5 * ((rho1 + rho2) - (rho1 - rho2) * np.tanh(2 * (r - R_eq) / t1))
        h = lambda z: 0.5 * (1 + 1 * np.tanh(2 * z / t2))
        r = np.sqrt(xi**2 + (zi - zi_c)**2)
        z = zi - zi_0
        out = g(r) * h(z)
        return out

    def fit(self, x_data, density_data):
        """
        Fit the hyperbolic tangent model to density data.
        
        Parameters:
        x_data: tuple (xi_array, zi_array) of coordinates
        density_data: array of density values
        
        Returns:
        self: for method chaining
        """
        self.params, self.covariance = curve_fit(
            self._fitting_function,
            x_data,
            density_data,
            p0=self.params,
            maxfev=1000000
        )
        return self

    def evaluate(self, x):
        """
        Evaluate the fitted function at point x.
        
        Parameters:
        x: tuple (xi, zi) of coordinates
        
        Returns:
        float: density value at the given point
        """
        if self.params is None:
            raise ValueError("Model must be fitted before evaluation")
        return self._fitting_function(
            x,
            self.params[0],
            self.params[1],
            self.params[2],
            self.params[3],
            self.params[4],
            self.params[5],
            self.params[6]
        )

    def compute_isoline(self, scale_factor=0.95):
        """
        Compute the iso-surface line for the density field.
        
        Parameters:
        scale_factor: factor to scale the radius for visualization (default: 0.95)
        
        Returns:
        tuple: (circle_xi, circle_zi, wall_line_xi, wall_line_zi)
        """
        if self.params is None:
            raise ValueError("Model must be fitted before computing isoline")

        R = scale_factor * self.params[2]  # R_eq
        Zcenter = self.params[3]  # zi_c
        Zwall = self.params[4]  # zi_0

        xi_wall = np.sqrt(R**2 - (Zwall - Zcenter)**2)
        alpha_inf = np.arctan((Zwall - Zcenter) / xi_wall)
        alpha = np.linspace(alpha_inf, np.pi / 2, 100)

        Xicenter = 1.0
        circle_xi = Xicenter + R * np.cos(alpha)
        circle_zi = Zcenter + R * np.sin(alpha)

        wall_line_xi = np.linspace(Xicenter, xi_wall, 100)
        wall_line_zi = np.ones((len(wall_line_xi))) * Zwall

        return circle_xi, circle_zi, wall_line_xi, wall_line_zi

    def compute_contact_angle(self):
        """
        Calculate the contact angle from the fitted parameters.
        
        Parameters:
        wall_height: height of the wall
        
        Returns:
        float: contact angle in degrees
        """
        if self.params is None:
            raise ValueError("Model must be fitted before computing contact angle")

        R_eq = self.params[2]
        zita_c = self.params[3]
        zita_wall = self.params[4]

        xi_cross = np.sqrt(R_eq**2 - (zita_wall - zita_c)**2)
        theta = (np.pi / 2 - np.arctan((zita_wall - zita_c) / xi_cross)) * 180 / np.pi

        return theta

    def get_parameters(self):
        """
        Get the fitted parameters with their names.
        
        Returns:
        dict: Dictionary of parameter names and values
        """
        if self.params is None:
            raise ValueError("Model must be fitted before getting parameters")

        param_names = ["rho1", "rho2", "R_eq", "zi_c", "zi_0", "t1", "t2"]
        return {name: value for name, value in zip(param_names, self.params)}

    def get_parameter_strings(self):
        """
        Get formatted strings of parameters for logging.
        
        Returns:
        list: List of formatted parameter strings
        """
        if self.params is None:
            raise ValueError("Model must be fitted before getting parameter strings")

        param_names = ["rho1", "rho2", "R_eq", "zi_c", "zi_0", "t1", "t2"]
        return [f"{name}:{value}\n" for name, value in zip(param_names, self.params)]
                                                                                         