
from hydroangleanalyzer.visualization_statistics_angles import Sliced_Trajectory_Analyzer

directories = ["result_dump_traj_12k_reduce_sliced/", "result_dump_traj_4k_reduce_sliced/"]
analyzer = Sliced_Trajectory_Analyzer(directories)
analyzer.analyze()
analyzer.plot_comparison_median_alfas("comparison_median_alfas.png")


""" to plot surface atoms of water droplet from sliced method

In [3]: if __name__ == "__main__":
   ...:     from hydroangleanalyzer.parser import DumpParser, Dump_WaterMoleculeFinder, DumpParse_wall
   ...:     from hydroangleanalyzer.contact_angle_method.sliced_method import ContactAngle_sliced
   ...:     import numpy as np
   ...:     from hydroangleanalyzer.visualization_statistics_angles import Droplet_sliced_Plotter
   ...:     filename = "../HydroAngleAnalyzer/tests/trajectories/traj_10_3_330w_nve_4k_reajust.lammpstrj"
   ...:     wat_find = Dump_WaterMoleculeFinder(filename, particle_type_wall={3}, oxygen_type=1, hydrogen_type=2)
   ...:     oxygen_indices = wat_find.get_water_oxygen_ids(num_frame=0)
   ...: 
   ...:     parser = DumpParser(in_path=filename)
   ...:     oxygen_position = parser.parse(num_frame=10, indices=oxygen_indices)
   ...: 
   ...:     coord_wall = DumpParse_wall(filename, particule_liquid_type={2,1})
   ...:     wall_coords = coord_wall.parse(num_frame=1)
   ...: 
   ...:     processor = ContactAngle_sliced(
   ...:         o_coords=oxygen_position,
   ...:         o_center_geom=np.mean(oxygen_position, axis=0),
   ...:         type_model='cylinder_y',
   ...:         delta_cylinder=5,
   ...:         max_dist=100,
   ...:         width_cylinder=21
   ...:     )
   ...: 
   ...:     list_alfas, array_surfaces, array_popt = processor.predict_contact_angle()
   ...:     print("Mean contact angles:", list_alfas)
   ...: 
   ...:     plotter = Droplet_sliced_Plotter(center=True, show_wall=True, molecule_view=True)
   ...:     for i in range(len(array_surfaces)):
   ...:         plotter.plot_surface_points(
   ...:             oxygen_position,
   ...:             np.array([array_surfaces[i]]),
   ...:             array_popt[i],
   ...:             wall_coords,
   ...:             f"Gsurface_plot_frame_{i}_fromlib.png",
   ...:             np.mean(oxygen_position[:, 1]),
   ...:             None
   ...:         )
   ...: 
""