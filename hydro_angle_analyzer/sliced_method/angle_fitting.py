#angle fit
import numpy as np
from .surface_defined import SurfaceDefinition
import numpy as np
from scipy.optimize import curve_fit

class ContactAnglePredictor:
    def __init__(self, o_coords, delta_gamma, max_dist, o_center_geom, z_wall, y_width, delta_y_axis, type='masspain'):
        self.o_coords = o_coords
        self.delta_gamma = delta_gamma
        self.max_dist = max_dist
        self.o_center_geom = o_center_geom
        self.z_wall = z_wall
        self.y_width = y_width
        self.delta_y_axis = delta_y_axis
        self.type = type

    def calculate_y_axis_list(self):
        if self.type == 'masspain':
            return np.arange(0, self.y_width, self.delta_y_axis)
        elif self.type == 'spherical':
            return [self.o_center_geom[1]]*int(180 / self.delta_gamma)
    def calculate_gammas_list(self):
        if self.type == 'masspain':
            return [0]*len(np.arange(0, self.y_width, self.delta_y_axis))
        elif self.type == 'spherical':
            return np.linspace(0, 180, int(180 / self.delta_gamma))
    def surface_definition(self, v_gamma):
        delta_angle = 4 if self.type == 'masspain' else 5
        surface_def = SurfaceDefinition(self.o_coords, delta_angle, self.max_dist, self.o_center_geom,  v_gamma)
        list_rr, list_xz = surface_def.analyze_lines()
        return np.array(list_xz), np.array(list_rr)

    def separate_surface_data(self, surf, limit_med):
        return surf[(surf[:, 1] > limit_med)]

    def fit_circle(self, X_data, Y_data, initial_guess, bounds):
        lower_bounds, upper_bounds = zip(*bounds)
        popt, _ = curve_fit(self.circle_equation, (X_data, Y_data), np.zeros_like(X_data), p0=initial_guess, bounds=(lower_bounds, upper_bounds))
        return popt

    def find_intersection(self, popt, y_line):
        Xs, Ys, R = popt
        delta_y = y_line - Ys
        discriminant = R**2 - delta_y**2
        if discriminant < 0:
            return None
        else:
            x_intersections = Xs + np.array([-1, 1]) * np.sqrt(discriminant)
            x_intersection = x_intersections[0]
            dx = x_intersection - Xs
            dy = y_line - Ys
            tangent_slope = -dx / dy
            tangent_angle = np.arctan(tangent_slope)
            return np.degrees(tangent_angle)
    
    def circle_equation(self, xy_data, x_center, z_center, radius):
        X_data, Y_data = xy_data
        return np.sqrt((X_data - x_center)**2 + (Y_data - z_center)**2) - radius

    def predict_contact_angle(self):
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
            surf_line = self.separate_surface_data(surf, limit_med)
            X_data = surf_line[:, 0]
            Y_data = surf_line[:, 1]
            mean_rr = np.mean(list_rr[:, 0])
            initial_guess = [self.o_center_geom[0], self.o_center_geom[2], mean_rr]
            bound = [
                (-self.max_dist - self.o_center_geom[0], self.o_center_geom[0] + self.max_dist),
                (-self.max_dist + self.o_center_geom[2], self.max_dist + self.o_center_geom[2]),
                (0, 10 + mean_rr)
            ]
            popt = self.fit_circle(X_data, Y_data, initial_guess, bound)
            array_popt.append(popt)
            angle = self.find_intersection(popt, self.z_wall)
            if angle is not None:
                list_alfas.append(np.abs(angle))

        return list_alfas, array_surfaces, array_popt