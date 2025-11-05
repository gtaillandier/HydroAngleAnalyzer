import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import os
import re
import numpy as np
from .base_trajectory_analyzer import BaseTrajectoryAnalyzer


class BinningTrajectoryAnalyzer(BaseTrajectoryAnalyzer):
    """
    Analyzer for binned trajectory data using circular segment calculations.
    """
    
    def _initialize_data_structure(self):
        """Initialize data structure for binning analysis."""
        for directory in self.directories:
            self.data[directory] = {
                "R_eq": [],
                "zi_c": [],
                "zi_0": [],
                "contact_angles": [],
                "surface_areas": [],
            }
    
    def get_method_name(self):
        """Return method name."""
        return "Binning Analysis"
    
    @staticmethod
    def circular_segment_area(R, z_center, z_cut):
        """
        Compute the area of a circular segment (cap) of a circle with radius R.
        Handles any cut position above or below the center.
        
        Parameters
        ----------
        R : float
            Radius of the circle.
        z_center : float
            z-coordinate of the circle center.
        z_cut : float
            z-coordinate of the cut line.
            
        Returns
        -------
        float
            Area of the circular segment.
        """
        h = (z_center + R) - z_cut  # height of the cap
        if h <= 0:
            return 0.0
        if h >= 2 * R:
            return np.pi * R**2
        if h <= R:
            return R**2 * np.arccos((R - h) / R) - (R - h) * np.sqrt(2 * R * h - h**2)
        else:
            h_small = 2 * R - h
            return np.pi * R**2 - (R**2 * np.arccos((R - h_small) / R) - 
                                   (R - h_small) * np.sqrt(2 * R * h_small - h_small**2))
    
    def read_data(self):
        """Read and parse data from log files in each directory."""
        for directory in self.directories:
            log_path = os.path.join(directory, "log_data.txt")
            if os.path.isfile(log_path):
                with open(log_path, "r") as f:
                    text = f.read()
                R_eq = float(re.search(r"R_eq:([0-9\.\-eE]+)", text).group(1))
                zi_c = float(re.search(r"zi_c:([0-9\.\-eE]+)", text).group(1))
                zi_0 = float(re.search(r"zi_0:([0-9\.\-eE]+)", text).group(1))
                angle = float(re.search(r"Contact angle:([0-9\.\-eE]+)", text).group(1))
                A_seg = self.circular_segment_area(R_eq, zi_c, zi_0)
                self.data[directory]["R_eq"].append(R_eq)
                self.data[directory]["zi_c"].append(zi_c)
                self.data[directory]["zi_0"].append(zi_0)
                self.data[directory]["contact_angles"].append(angle)
                self.data[directory]["surface_areas"].append(A_seg)
    
    def get_surface_areas(self, directory):
        """Get surface areas for a directory."""
        return np.array(self.data[directory]["surface_areas"])
    
    def get_contact_angles(self, directory):
        """Get contact angles for a directory."""
        return np.array(self.data[directory]["contact_angles"])
# Example usage:
# analyzer = Binning_Trajectory_Analyzer(directories=["./result_dump_1", "./result_dump_2"])
# analyzer.analyze()
# analyzer.plot_mean_angle_vs_surface(save_path="mean_angle_vs_surface.png")