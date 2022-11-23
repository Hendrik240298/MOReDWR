import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.linalg
import scipy.interpolate
from scipy.sparse import coo_matrix, bmat
import matplotlib.pyplot as plt
import os
import time
import sys
#from iPOD import iPOD
import imageio

PLOTTING = True
INTERPOLATION_TYPE = "nearest"  # "linear", "cubic"
CASE = ""  # "rotating_circle"
OUTPUT_PATH = "../../../Data/2D/BangerthGeigerRannacher/"
cycle = "cycle=4"
SAVE_PATH = "../../../Data/2D/BangerthGeigerRannacher/" + cycle + "/output_ROM/"

# print(os.listdir(OUTPUT_PATH))

print(f"\n{'-'*12}\n| {cycle}: |\n{'-'*12}\n")
# NO BC
[data, row, column] = np.loadtxt(OUTPUT_PATH + cycle + "/matrix_no_bc.txt")
matrix_no_bc = scipy.sparse.csr_matrix(
    (data, (row.astype(int), column.astype(int))))
A = matrix_no_bc[:, :].toarray()

"""
[data, row, column] = np.loadtxt(OUTPUT_PATH + cycle + "/mass_matrix_no_bc.txt")
mass_matrix_no_bc = scipy.sparse.csr_matrix(
    (data, (row.astype(int), column.astype(int))))
"""

rhs_no_bc = []
for f in sorted([f for f in os.listdir(OUTPUT_PATH + cycle)
                if "dual" not in f and "rhs_no_bc" in f]):
    rhs_no_bc.append(np.loadtxt(OUTPUT_PATH + cycle + "/" + f))

"""
dual_rhs_no_bc = []
for f in sorted([f for f in os.listdir(
        OUTPUT_PATH + cycle) if "dual_rhs_no_bc" in f]):
    dual_rhs_no_bc.append(np.loadtxt(OUTPUT_PATH + cycle + "/" + f))
"""

initial_solution = np.loadtxt(OUTPUT_PATH + cycle + "/initial_solution.txt")

# NO boundary_ids, since we have only homogeneous Neumann boundary conditions for the wave equation which represent reflecting boundaries
n_space_dofs = int(matrix_no_bc.shape[0]/2)

# %% applying BC to primal matrix
primal_matrix = matrix_no_bc.tocsr()
for row in range(n_space_dofs):
    for col in primal_matrix.getrow(row).nonzero()[1]:
        primal_matrix[row, col] = 1. if row == col else 0.

# %% coordinates
coordinates_x = np.loadtxt(OUTPUT_PATH + cycle + "/coordinates_x.txt")
list_coordinates_t = []
for f in sorted([f for f in os.listdir(
        OUTPUT_PATH + cycle) if "coordinates_t" in f]):
    list_coordinates_t.append(np.loadtxt(OUTPUT_PATH + cycle + "/" + f))
n_slabs = len(list_coordinates_t)
coordinates_t = np.hstack(list_coordinates_t)
coordinates = np.vstack((
    np.tensordot(coordinates_t, np.ones_like(coordinates_x), 0).flatten(),
    np.tensordot(np.ones_like(coordinates_t), coordinates_x, 0).flatten()
)).T
n_dofs = {"space": coordinates_x.shape[1], "time": coordinates_t.shape[0]}

# ------------
# %% primal FOM solve
start_execution = time.time()
last_primal_solution = np.zeros_like(rhs_no_bc[0])
last_primal_solution[n_space_dofs:] = initial_solution[:]
primal_solutions = []
for i in range(n_slabs):
    # creating primal rhs and applying BC to it
    primal_rhs = rhs_no_bc[i].copy()
    for i in range(n_space_dofs):
        primal_rhs[i] = last_primal_solution[i + n_space_dofs]

    primal_solutions.append(
        scipy.sparse.linalg.spsolve(primal_matrix, primal_rhs))

    last_primal_solution = primal_solutions[-1]
end_execution = time.time()
execution_time_FOM = end_execution - start_execution
print("FOM time:   " + str(execution_time_FOM))


# plotting the primal FOM solution
if PLOTTING:
    def primal_gif(primal_solution):
        grid_x, grid_y = np.mgrid[-1:1:150j, -1:1:150j]
        displacement_grid = scipy.interpolate.griddata(
            coordinates_x.T, primal_solution[0:n_dofs["space"]], (grid_x, grid_y), method=INTERPOLATION_TYPE)
        # fig, _ = plt.subplots(figsize=(15,15))

        fig, (ax1, ax2) = plt.subplots(2, figsize=(30, 30))

        ax1.set_title(f"Primal displacement (ref={cycle.split('=')[1]})")
        # ,vmin=-0.00025,vmax=0.00025)
        im1 = ax1.imshow(displacement_grid.T,
                         extent=(-1, 1, -1, 1), origin='lower')
        # ax1.set_clim([-0.00025, 0.00025])
        ax1.set_xlabel("$y$")
        ax1.set_ylabel("$x$")
        # ax1.colorbar()
        # plt.colorbar(im1,ax=ax1)

        velocity_grid = scipy.interpolate.griddata(
            coordinates_x.T, primal_solution[n_dofs["space"]:2 * n_dofs["space"]], (grid_x, grid_y), method=INTERPOLATION_TYPE)
        ax2.set_title(f"Primal velocity (ref={cycle.split('=')[1]})")
        # ,vmin=-0.00025,vmax=0.00025)
        im2 = ax2.imshow(velocity_grid.T,
                         extent=(-1, 1, -1, 1), origin='lower')
        # ax2.set_clim(-0.00025, 0.00025)
        ax2.set_xlabel("$y$")
        ax2.set_ylabel("$x$")
        # ax2.set_colorbar()
        # plt.colorbar(im2,ax=ax2)
        # plt.show()

        fig.canvas.draw()       # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        return image

    imageio.mimsave('./primal_solution.gif', [primal_gif(primal_solution)
                    for primal_solution in primal_solutions], fps=3)

print("\n\nTO DO @Hendrik")
