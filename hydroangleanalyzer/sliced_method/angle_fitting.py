import numpy as np
from .surface_defined import SurfaceDefinition
from scipy.optimize import curve_fit

class ContactAnglePredictor:
    def __init__(self, o_coords, delta_gamma, max_dist, o_center_geom, z_wall, y_width, delta_y_axis,  limit_dist_wall, type='masspain'):
        """
        Initialize the ContactAnglePredictor.

        Args:
            o_coords (array): Coordinates of oxygen atoms.
            delta_gamma (float): Angular step size for spherical calculations.
            max_dist (float): Maximum distance for surface analysis.
            o_center_geom (array): Geometric center of the system.
            z_wall (float): Z-coordinate of the wall.
            y_width (float): Width of the Y-axis range.
            delta_y_axis (float): Step size for Y-axis calculations.
            type (str): Type of analysis ('masspain' or 'spherical').
        """
        self.o_coords = o_coords
        self.delta_gamma = delta_gamma
        self.max_dist = max_dist
        self.o_center_geom = o_center_geom
        self.z_wall = z_wall
        self.y_width = y_width
        self.delta_y_axis = delta_y_axis
        self.limit_dist_wall =  limit_dist_wall
        self.type = type

    def calculate_y_axis_list(self):
        """
        Calculate the Y-axis positions based on the type of analysis.

        Returns:
            list: Y-axis positions.
        """
        if self.type == 'masspain':
            return np.arange(0, self.y_width, self.delta_y_axis)
        elif self.type == 'spherical':
            return [self.o_center_geom[1]] * int(180 / self.delta_gamma)

    def calculate_gammas_list(self):
        """
        Calculate the gamma values based on the type of analysis.

        Returns:
            list: Gamma values.
        """
        if self.type == 'masspain':
            return [0] * len(np.arange(0, self.y_width, self.delta_y_axis))
        elif self.type == 'spherical':
            return np.linspace(0, 180, int(180 / self.delta_gamma))

    def surface_definition(self, v_gamma):
        """
        Define the surface based on the input gamma value.

        Args:
            v_gamma (float): Gamma value for surface definition.

        Returns:
            tuple: Arrays of XZ surface and radial distances.
        """
        delta_angle = 4 if self.type == 'masspain' else 5
        surface_def = SurfaceDefinition(self.o_coords, delta_angle, self.max_dist, self.o_center_geom, v_gamma)
        list_rr, list_xz = surface_def.analyze_lines()
        return np.array(list_xz), np.array(list_rr)

    def separate_surface_data(self, surf, limit_med):
        """
        Separate surface data based on a median limit.

        Args:
            surf (array): Surface data.
            limit_med (float): Median limit for filtering.

        Returns:
            array: Filtered surface data.
        """
        return surf[(surf[:, 1] > limit_med)]

    def fit_circle(self, X_data, Y_data, initial_guess):
        """
        Fit a circle to the surface data.

        Args:
            X_data (array): X-coordinate data.
            Y_data (array): Y-coordinate data.
            initial_guess (list): Initial guess for circle parameters.
            
        Returns:
            array: Optimal circle parameters.
        """
        popt, _ = curve_fit(self.circle_equation, (X_data, Y_data), np.zeros_like(X_data), p0=initial_guess)
        return popt
    
    def fit_circle_with_baseline_cutoff(self, X_data, Y_data, initial_guess):
        """
        Fit a circle to the surface data with a baseline cutoff constraint.
        This creates a model of a circle that's cut by a horizontal line at z_wall.
        
        Args:
            X_data (array): X-coordinate data.
            Y_data (array): Y-coordinate data.
            initial_guess (list): Initial guess for circle parameters [x_center, z_center, radius].
            
        Returns:
            array: Optimal circle parameters [x_center, z_center, radius].
        """
        # Define a model for a circle cut by a horizontal line
        def truncated_circle_model(params, x):
            x_center, z_center, radius = params
            
            # Calculate the y-coordinate on the circle for each x value
            # For a circle: (x-x_center)² + (y-z_center)² = radius²
            # Therefore, y = z_center ± sqrt(radius² - (x-x_center)²)
            # We want the upper part of the circle, so use the + solution
            
            # Handle points outside the circle's x-range
            inside_circle = np.abs(x - x_center) <= radius
            y_values = np.full_like(x, np.nan)
            
            # Only calculate for valid x-positions
            valid_x = x[inside_circle]
            y_valid = z_center + np.sqrt(radius**2 - (valid_x - x_center)**2)
            
            # Apply the truncation at z_wall (only keep points above z_wall)
            y_valid = np.maximum(y_valid, self.z_wall)
            y_values[inside_circle] = y_valid
            
            return y_values
        
        # Define error function for optimization
        def error_function(params, x_data, y_data):
            predicted_y = truncated_circle_model(params, x_data)
            # Filter out NaN values (points outside the circle's valid x-range)
            valid_mask = ~np.isnan(predicted_y)
            if not np.any(valid_mask):
                return np.full_like(y_data, 1e6)  # Return large error if all predictions invalid
            
            # Calculate errors only for valid predictions
            error = y_data[valid_mask] - predicted_y[valid_mask]
            return error
        
        # Use robust optimization to find the best parameters
        from scipy.optimize import least_squares
        result = least_squares(
            error_function, 
            initial_guess, 
            args=(X_data, Y_data),
            loss='soft_l1',  # Robust loss function to handle outliers
            method='trf'     # Trust Region Reflective algorithm works well for this
        )
        return result.x
    def find_intersection(self, popt, y_line):
        """
        Find the intersection of the circle with a horizontal line.

        Args:
            popt (array): Circle parameters (center X, center Z, radius).
            y_line (float): Y-coordinate of the line.

        Returns:
            float: Angle of intersection in degrees, or None if no intersection exists.
        """
        Xs, Ys, R = popt
        delta_y = y_line - Ys
        discriminant = R**2 - delta_y**2
        if discriminant < 0:
            return None
        else:
            theta = np.arccos(delta_y / R)
            contact_angle = np.degrees(theta)
            return contact_angle
    def circle_equation(self, xy_data, x_center, z_center, radius):
        """
        Equation of a circle used for curve fitting.

        Args:
            xy_data (tuple): Tuple of X and Y data.
            x_center (float): X-coordinate of the circle center.
            z_center (float): Z-coordinate of the circle center.
            radius (float): Radius of the circle.

        Returns:
            array: Difference between calculated and actual radius values.
        """
        X_data, Y_data = xy_data
        return np.sqrt((X_data - x_center)**2 + (Y_data - z_center)**2) - radius

    def predict_contact_angle(self):
        """
        Predict contact angles based on surface analysis.

        Returns:
            tuple: Lists of contact angles, surfaces, and circle parameters.
        """
        gammas = self.calculate_gammas_list()

        y_axis_list = self.calculate_y_axis_list()
        #limit_med = 9.5 if self.type == 'masspain' else 8
        list_alfas = []
        array_surfaces = []
        array_popt = []
        counter = 0

        for value_gamma in gammas:
            self.o_center_geom[1] = y_axis_list[counter]
            counter += 1
            surf, list_rr = self.surface_definition(value_gamma)
            array_surfaces.append(surf)
            min_drop = np.min(surf[:,1])
            surf_line = self.separate_surface_data(surf, min_drop+2)#self.limit_dist_wall)
            X_data = surf_line[:, 0]
            Y_data = surf_line[:, 1]
            mean_rr = np.mean(list_rr[:, 0])
            initial_guess = [self.o_center_geom[0], self.o_center_geom[2], mean_rr]
            popt = self.fit_circle(X_data, Y_data, initial_guess)
            array_popt.append(popt)
            angle = self.find_intersection(popt,min_drop+2) # self.z_wall)
            if angle is not None:
                list_alfas.append(angle)

        return list_alfas, array_surfaces, array_popt
    
    def predict_contact_angle_nobound(self):
        """
        Predict contact angles based on surface analysis.

        Returns:
            tuple: Lists of contact angles, surfaces, and circle parameters.
        """
        gammas = self.calculate_gammas_list()
        y_axis_list = self.calculate_y_axis_list()
        limit_med = 9.5 if self.type == 'masspain' else 14
        list_alfas = []
        array_surfaces = []
        array_popt = []
        counter = 0

        for value_gamma in gammas:
            self.o_center_geom[1] = y_axis_list[counter]
            counter += 1
            surf, list_rr = self.surface_definition(value_gamma)
            array_surfaces.append(surf)
            surf_line = self.separate_surface_data(surf, limit_med)
            X_data = surf_line[:, 0]
            Y_data = surf_line[:, 1]
            mean_rr = np.mean(list_rr[:, 0])
            initial_guess = [self.o_center_geom[0], self.o_center_geom[2], mean_rr]
            
            popt = self.fit_circle(X_data, Y_data, initial_guess)
            array_popt.append(popt)
            angle = self.find_intersection(popt, self.z_wall)
            if angle is not None:
                list_alfas.append(angle)

        return list_alfas, array_surfaces, array_popt
    def predict_contact_angle_with_truncated_circle(self):
        """
        Predict contact angles using a truncated circle model that accounts for the baseline.
        
        Returns:
            tuple: Lists of contact angles, surfaces, and circle parameters.
        """
        gammas = self.calculate_gammas_list()
        y_axis_list = self.calculate_y_axis_list()
        limit_med = 9.5 if self.type == 'masspain' else 8
        list_alfas = []
        array_surfaces = []
        array_popt = []
        counter = 0

        for value_gamma in gammas:
            self.o_center_geom[1] = y_axis_list[counter]
            counter += 1
            surf, list_rr = self.surface_definition(value_gamma)
            array_surfaces.append(surf)
            
            # Initial filtering to remove obvious outliers
            surf_line = self.separate_surface_data(surf, limit_med)
            X_data = surf_line[:, 0]
            Y_data = surf_line[:, 1]
            mean_rr = np.mean(list_rr[:, 0])
            initial_guess = [self.o_center_geom[0], self.o_center_geom[2], mean_rr]
            
            # Use the new truncated circle fitting function
            popt = self.fit_circle_with_baseline_cutoff(X_data, Y_data, initial_guess)
            array_popt.append(popt)
            
            # Calculate the contact angle at the intersection with z_wall
            angle = self.find_intersection(popt, self.z_wall)
            if angle is not None:
                list_alfas.append(angle)

        return list_alfas, array_surfaces, array_popt