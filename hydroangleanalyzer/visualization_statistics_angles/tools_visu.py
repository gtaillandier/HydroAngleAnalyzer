import matplotlib.pyplot as plt
import numpy as np

def plot_surface_file(file_path):
    """Reads surface data from a text file."""
    data = np.loadtxt(file_path)
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    return x, y
def plot_slice(x, y):
    """Plots a 2D slice of the surface."""
    plt.figure()
    plt.plot(x, y, label='Surface Slice')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('2D Slice of Fitted Surface')
    plt.legend()
    plt.grid()
    plt.show()

def visualize_surface_with_points(surface_file, points):
    """Visualizes the fitted surface along with the data points."""
    x_surf, y_surf, z_surf = read_surface_file(surface_file)
    x_points, y_points, z_points = points[:, 0], points[:, 1], points[:, 2]
    
    plot_surface_and_points(x_surf, y_surf, z_surf, x_points, y_points, z_points)   
# Example usage:
# surface_file = 'path_to_surface_file.txt'
# points = np.array([[x1, y1, z1], [x2, y2, z2], ...])
# visualize_surface_with_points(surface_file, points)   