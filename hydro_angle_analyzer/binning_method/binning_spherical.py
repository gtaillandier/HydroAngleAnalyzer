import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit
from ovito.io import import_file, export_file
from ovito.modifiers import (SelectTypeModifier, DeleteSelectedModifier, ComputePropertyModifier)
import copy

# Use 'Agg' backend for matplotlib to save figures without displaying them
matplotlib.use('Agg')

# Simulation parameters
ZITA = 0.4  # Parameter zita
TEMPERATURE = 0.8  # Reduced temperature

# Input and output paths
in_dir = "/home/gtaillandier/Documents/contact_angle_lammps/edocolad/scripts_manneback/data_tests"  # Input directory
in_file = "traj_10_3_330w.lammpstrj"  # Input file name
out_dir =  "../output_binning/zita_{}-T_{}-particles_".format(ZITA, TEMPERATURE)  # Output directory
in_path = in_dir + in_file  # Full input path
frames_number = 300  # Number of frames to process

# Binning parameters for cylindrical coordinates
xi_0, xi_f, nbins_xi = 0, 50.0, 30  # Range and number of bins for xi
zi_0, zi_f, nbins_zi = 0.0, 50.0, 30  # Range and number of bins for zi

# Initial guess for fitting parameters
guess_fitting = [1e-3, 3e-2, 40.0, 20.0, 4.0, 1.0, 1.0]

# Define the fitting function based on a hyperbolic tangent model
def fitting_function(x, rho1, rho2, R_eq, zi_c, zi_0, t1, t2):
    xi, zi = x[0], x[1]
    g = lambda r: 0.5 * ((rho1 + rho2) - (rho1 - rho2) * np.tanh(2 * (r - R_eq) / t1))
    h = lambda z: 0.5 * (1 + np.tanh(2 * z / t2))
    r = np.sqrt(xi**2 + (zi - zi_c)**2)
    z = zi - zi_0
    out = g(r) * h(z)
    return out

# Define an alternative fitting function based on an ellipsoidal model
def ellipsoid_fitting_function(x, rho1, rho2, a, b, zi_c, zi_0, t1, t2):
    xi, zi = x[0], x[1]
    g = lambda r: 0.5 * ((rho1 + rho2) - (rho1 - rho2) * np.tanh(2 * (r - 1) / t1))
    h = lambda z: 0.5 * (1 + np.tanh(2 * z / t2))
    r = np.sqrt((xi/a)**2 + ((zi - zi_c)/b)**2)
    z = zi - zi_0
    out = g(r) * h(z)
    return out

# Load the simulation data using OVITO
def load_dump_ovito(in_path):
    pipeline = import_file(in_path)
    pipeline.modifiers.append(SelectTypeModifier(property='Particle Type', types={2, 3}))
    pipeline.modifiers.append(DeleteSelectedModifier())
    pipeline.modifiers.append(ComputePropertyModifier(expressions=['1'], output_property='Unity'))
    return pipeline

# Convert Cartesian coordinates to cylindrical coordinates
def return_cylindrical_coord_pars(pipeline, frames_number):
    frames_tot = pipeline.source.num_frames
    last_frames = frames_number
    frame_list = [i for i in range(frames_tot - last_frames, frames_tot, 1)]
    xi_par = np.array([])
    zi_par = np.array([])
    for frame in frame_list:
        data = pipeline.compute(frame)
        X_par = np.asarray(data.particles["Position"])
        dim = len(X_par[0, :])
        X_cm = [(X_par[:, i]).sum() / len(X_par[:, i]) for i in range(dim)]
        X_0 = [X_par[:, i] - X_cm[i] * (i < 2) for i in range(dim)]
        xi_par_frame = np.sqrt(X_0[0]**2 + X_0[1]**2)
        zi_par_frame = X_0[2]
        xi_par = np.concatenate((xi_par, xi_par_frame))
        zi_par = np.concatenate((zi_par, zi_par_frame))
        if frame % 10 == 0:
            print(f"frame: {frame}/{frames_tot}")
            print(X_cm)
    print("\nxi range:\t({},{})".format(np.min(xi_par), np.max(xi_par)))
    print("zi range:\t({},{})".format(np.min(zi_par), np.max(zi_par)))
    return xi_par, zi_par, len(frame_list)

# Bin the particle data into a 2D grid to compute the density field
def binning(xi_par, zi_par, xi, zi, xi_cc, zi_cc, len_frames):
    print("Binning ...")
    dxi = xi[1] - xi[0]
    dzi = zi[1] - zi[0]
    rho_cc = np.zeros((len(xi_cc), len(zi_cc)))
    xi_par_0, zi_par_0 = copy.deepcopy(xi_par), copy.deepcopy(zi_par)
    for i in range(len(xi_cc)):
        if i % 1 == 0:
            print(f"Advancement: {100 * i / (len(xi) - 1)}%")
        dV = 2 * np.pi * (xi_cc[i]) * dxi * dzi
        for j in range(len(zi_cc)):
            where = (xi_par_0 > xi[i]) * (xi_par_0 < xi[i + 1]) * (zi_par_0 > zi[j]) * (zi_par_0 < zi[j + 1])
            count_i = where.sum()
            rho_cc[i, j] = count_i / dV
            in_this_bin_indx = np.nonzero(where)
            xi_par_0 = np.delete(xi_par_0, in_this_bin_indx)
            zi_par_0 = np.delete(zi_par_0, in_this_bin_indx)
    rho_cc /= len_frames
    return rho_cc

# Plot the 2D density field
def plotting_f_2d(xi_cc, zi_cc, rho_cc, out_dir, clevels=20, scale=0.75, name="pic.png", close=True):
    fig = plt.figure(dpi=300, figsize=(4 * scale, 3 * scale))
    plt.contourf(xi_cc, zi_cc, np.transpose(rho_cc), levels=clevels, cmap="jet")
    plt.colorbar()
    plt.savefig(out_dir + "/" + name)
    if close:
        plt.close()

# Plot 1D lines (e.g., iso-surface lines)
def plotting_f_1d(circle_xi, circle_zi, out_dir, scale=0.75, name="pic.png", close=True):
    plt.plot(circle_xi, circle_zi, "--", color="black")
    plt.savefig(out_dir + "/" + name)
    if close:
        plt.close()

# Fit the density data to the theoretical model using curve_fit
def fitting(f, xi_cc, zi_cc, rho_cc, guess_fitting=None):
    msh_zi_cc_grid, msh_xi_cc_grid = np.meshgrid(zi_cc, xi_cc)
    msh_zi_cc = msh_zi_cc_grid.reshape((len(xi_cc) * len(zi_cc)), order="F")
    msh_xi_cc = msh_xi_cc_grid.reshape((len(xi_cc) * len(zi_cc)), order="F")
    msh_rho_cc = rho_cc.reshape((len(xi_cc) * len(zi_cc)), order="F")
    x_data = (msh_xi_cc, msh_zi_cc)
    rho_data = msh_rho_cc
    p0 = guess_fitting
    if p0 is None:
        parameters, covariance = curve_fit(f, x_data, rho_data, maxfev=1000000)
    else:
        parameters, covariance = curve_fit(f, x_data, rho_data, p0=p0, maxfev=1000000)
    fitted_function = lambda x: f(x, parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6])
    return fitted_function, parameters, covariance

# Evaluate the fitted function on a grid
def fit_on_grid(f, xi_cc, zi_cc):
    out_fitted = np.zeros((len(xi_cc), len(zi_cc)))
    for i in range(len(xi_cc)):
        for j in range(len(zi_cc)):
            out_fitted[i, j] = f([xi_cc[i], zi_cc[j]])
    return out_fitted

# Calculate the contact angle from the fitted parameters
def compute_contact_angle(fitted_param):
    R_eq = fitted_param[2]
    zita_c = fitted_param[3]
    zita_wall = fitted_param[4]
    xi_cross = np.sqrt(R_eq**2 - (zita_wall - zita_c)**2)
    theta = (np.pi / 2 - np.arctan((zita_wall - zita_c) / xi_cross)) * 180 / np.pi
    return theta

# Compute the iso-surface line for the density field
def compute_density_isoline(R, Zcenter, Zwall):
    xi_wall = np.sqrt(R**2 - (Zwall - Zcenter)**2)
    alpha_inf = np.arctan((Zwall - Zcenter) / xi_wall)
    alpha = np.linspace(alpha_inf, np.pi / 2, 100)
    Xicenter = 1.0
    circle_xi = Xicenter + R * np.cos(alpha)
    circle_zi = Zcenter + R * np.sin(alpha)
    wall_line_xi = np.linspace(Xicenter, xi_wall, 100)
    wall_line_zi = np.ones((len(wall_line_xi))) * Zwall
    return circle_xi, circle_zi, wall_line_xi, wall_line_zi

# Save the simulation parameters, fitted parameters, contact angle, and density field to a log file and a CSV file
def save_logfile(ZITA, TEMPERATURE, PARTICLES_NUMBER, out_dir, fp_string_list, theta, xi_cc, zi_cc, rho_cc):
    with open(out_dir + "log_data.txt", 'w') as f:
        f.write("Simulation parameters:\n")
        f.write("zita:{}\nreduced_temperature:{}\nparticles_number:{}\n".format(ZITA, TEMPERATURE, PARTICLES_NUMBER))
        f.write("Fitted parameters:\n")
        for i in range(len(fp_string_list)):
            f.write(fp_string_list[i])
        f.write("\n\nContact angle:{}".format(theta))
        msh_zi_cc_grid, msh_xi_cc_grid = np.meshgrid(zi_cc, xi_cc)
        msh_zi_cc = msh_zi_cc_grid.reshape((len(xi_cc) * len(zi_cc)), order="F")
        msh_xi_cc = msh_xi_cc_grid.reshape((len(xi_cc) * len(zi_cc)), order="F")
        msh_rho_cc = rho_cc.reshape((len(xi_cc) * len(zi_cc)), order="F")
        CSV = np.c_[msh_xi_cc, msh_zi_cc, msh_rho_cc]
        np.savetxt(out_dir + "rho_field.csv", CSV, delimiter=",", header="x_{},y_{},rho_{}".format(len(xi_cc), len(zi_cc), len(xi_cc) * len(zi_cc)))

# Main function to execute the entire post-processing workflow
def postprocess_script(ZITA, TEMPERATURE, in_path, out_dir, frames_number, xi_0, xi_f, nbins_xi, zi_0, zi_f, nbins_zi, guess_fitting):
    xi = np.linspace(xi_0, xi_f, nbins_xi)
    zi = np.linspace(zi_0, zi_f, nbins_zi)
    dxi = xi[1] - xi[0]
    dzi = zi[1] - zi[0]
    xi_cc = 0.5 * (xi[1:] + xi[:-1])
    zi_cc = 0.5 * (zi[1:] + zi[:-1])
    pipeline = load_dump_ovito(in_path)
    print("load finished")
    xi_par, zi_par, len_frames = return_cylindrical_coord_pars(pipeline, frames_number)
    PARTICLES_NUMBER = len(xi_par) / len_frames
    print("\nNumber of fluid particles:\t{}\n".format(PARTICLES_NUMBER))
    out_dir += str(int(PARTICLES_NUMBER)) + "/"
    os.makedirs(out_dir, exist_ok=True)
    rho_cc = binning(xi_par, zi_par, xi, zi, xi_cc, zi_cc, len_frames)
    fitted_function, fitted_param, covariance = fitting(fitting_function, xi_cc, zi_cc, rho_cc, guess_fitting=guess_fitting)
    print("\nFitted parameters:")
    parameter_list = ["rho1", "rho2", "R_eq", "zi_c", "zi_0", "t1", "t2"]
    fp_string_list = []
    for i in range(len(fitted_param)):
        item = parameter_list[i]
        fp_string_list.append(item + ":" + str(fitted_param[i]) + "\n")
    temp = ""
    for item in fp_string_list:
        temp += item
    print(temp)
    theta = compute_contact_angle(fitted_param)
    circle_xi, circle_zi, wall_line_xi, wall_line_zi = compute_density_isoline(R=0.95 * fitted_param[2], Zcenter=fitted_param[3], Zwall=fitted_param[4])
    rho_fitted = fit_on_grid(fitted_function, xi_cc, zi_cc)
    plotting_f_2d(xi_cc, zi_cc, rho_fitted, out_dir, clevels=20, scale=0.75, name="fitting.png")
    pic_name = "bin_plot.png"
    pic_scale = 0.75
    plotting_f_2d(xi_cc, zi_cc, rho_cc, out_dir, clevels=20, scale=pic_scale, name=pic_name, close=False)
    plotting_f_1d(circle_xi, circle_zi, out_dir, scale=pic_scale, name=pic_name, close=False)
    plotting_f_1d(wall_line_xi, wall_line_zi, out_dir, scale=pic_scale, name=pic_name, close=True)
    print("Contact angle:\t", theta)
    save_logfile(ZITA, TEMPERATURE, PARTICLES_NUMBER, out_dir, fp_string_list, theta, xi_cc, zi_cc, rho_cc)

# Execute the main function
postprocess_script(ZITA, TEMPERATURE, in_path, out_dir, frames_number, xi_0, xi_f, nbins_xi, zi_0, zi_f, nbins_zi, guess_fitting)