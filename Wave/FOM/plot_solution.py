import sys
import os
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt

assert len(sys.argv) == 2, "You need to enter the path to the solution!"
path = sys.argv[1]

def load_all_vectors(path, pattern):
  file_names = [f for f in os.listdir(path) if pattern in f]
  vectors = [np.loadtxt(f"{path}/{file_name}") for file_name in file_names]
  return np.hstack(vectors)

# space-time solution vectors
# displacement:
solution_u = load_all_vectors(path, "solution_u_")
# velocity:
solution_v = load_all_vectors(path, "solution_v_")

# coordinates
coordinates_x = np.loadtxt(path + "/coordinates_x.txt")
x_min, x_max = np.min(coordinates_x), np.max(coordinates_x)
coordinates_t = load_all_vectors(path, "coordinates_t_")
t_min, t_max = np.min(coordinates_t), np.max(coordinates_t)
coordinates = np.vstack((
    np.tensordot(coordinates_t, np.ones_like(coordinates_x), 0).flatten(),
    np.tensordot(np.ones_like(coordinates_t), coordinates_x, 0).flatten()
)).T
n_dofs = {"space": coordinates_x.shape[0], "time": coordinates_t.shape[0]}

#print("x_min:", x_min)
#print("x_max:", x_max)
#print("t_min:", t_min)
#print("t_max:", t_max)

grid_t, grid_x = np.mgrid[t_min:t_max:200j, x_min:x_max:200j]

# displacement
primal_grid = scipy.interpolate.griddata(
    coordinates, solution_u, (grid_t, grid_x), method="nearest")
plt.title("Displacement")
plt.imshow(primal_grid.T, extent=(t_min, t_max, x_min, x_max), origin='lower') #extent=(0, 4, 0, 1), 
plt.xlabel("$t$")
plt.ylabel("$x$")
plt.colorbar()
plt.savefig(path + "plot_u.png")
plt.clf()

# velocity
primal_grid = scipy.interpolate.griddata(
    coordinates, solution_v, (grid_t, grid_x), method="nearest")
plt.title("Velocity")
plt.imshow(primal_grid.T, extent=(t_min, t_max, x_min, x_max), origin='lower') #extent=(0, 4, 0, 1), 
plt.xlabel("$t$")
plt.ylabel("$x$")
plt.colorbar()
plt.savefig(path + "plot_v.png")

