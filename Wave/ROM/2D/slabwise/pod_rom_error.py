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

def save_vtk(file_name, solution, grid, cycle=None, time=None):
    lines = [
        "# vtk DataFile Version 3.0",
        "PDE SOLUTION",
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
	""
    ]

    if (cycle != None or time != None):
        lines.append(f"FIELD FieldData {(cycle != None) + (time != None)}")
        if (cycle != None):
            lines.append("CYCLE 1 1 int")
            lines.append(str(cycle))
        if (time != None):
            lines.append("TIME 1 1 double")
            lines.append(str(time))

    lines += grid

    for key, value in solution.items():
        lines.append(f"SCALARS {key} double 1")
        lines.append("LOOKUP_TABLE default")
        lines.append(" ".join(np.round(value, decimals=7).astype(np.double).astype(str)) + " ")

    with open(file_name, "w") as file:
        file.write("\n".join(lines))

PLOTTING = True
INTERPOLATION_TYPE = "nearest"  # "linear", "cubic"
CASE = ""  # "rotating_circle"
OUTPUT_PATH = "../../../Data/2D/BangerthGeigerRannacher/"
cycle = "cycle=5"
SAVE_PATH = "../../../Data/2D/BangerthGeigerRannacher/" + cycle + "/output_ROM/"

# creating grid and dof_matrix for vtk output
grid = []
with open(OUTPUT_PATH + cycle + "/solution00000.vtk", "r") as f:
    writing = False
    for line in f:
        if (not writing and line.startswith("POINTS")):
            writing = True
        if writing:
            grid.append(line.strip("\n"))
            if line.startswith("POINT_DATA"):
                break
#print(grid)
dof_vector = np.loadtxt(OUTPUT_PATH + cycle + "/dof.txt").astype(int)
dof_matrix = scipy.sparse.dok_matrix((dof_vector.shape[0],dof_vector.max()+1))
for i, j in enumerate(list(dof_vector)):
   dof_matrix[i,j] = 1.


print(f"\n{'-'*12}\n| {cycle}: |\n{'-'*12}\n")
# NO BC
[data, row, column] = np.loadtxt(OUTPUT_PATH + cycle + "/matrix_no_bc.txt")
matrix_no_bc = scipy.sparse.csr_matrix(
    (data, (row.astype(int), column.astype(int))))


[data, row, column] = np.loadtxt(OUTPUT_PATH + cycle + "/dual_matrix_no_bc.txt")
functional_matrix_no_bc = scipy.sparse.csr_matrix(
    (data, (row.astype(int), column.astype(int))))


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
print(last_primal_solution.shape)
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
print("Primal FOM time:   " + str(execution_time_FOM))

print("n_dofs[space] =", n_dofs["space"])
for i, primal_solution in enumerate(primal_solutions):
	save_vtk(OUTPUT_PATH + cycle + f"/py_solution{i:05}.vtk", {"displacement": dof_matrix.dot(primal_solution[0:n_dofs["space"]]), "velocity": dof_matrix.dot(primal_solution[n_dofs["space"]:2 * n_dofs["space"]])}, grid, cycle=i, time=list_coordinates_t[i][0])

# %% applying BC to dual matrix
dual_matrix = matrix_no_bc.T.tocsr()
for row in range(n_space_dofs):
    for col in dual_matrix.getrow(row).nonzero()[1]:
        dual_matrix[row, col] = 1. if row == col else 0.
        
# ------------
# %% dual FOM solve
start_execution = time.time()
last_dual_solution = np.zeros_like(rhs_no_bc[0])
# zero initial condition for dual problem
dual_solutions = []
for i in list(range(n_slabs))[::-1]:
    # creating dual rhs and applying BC to it
    dual_rhs = functional_matrix_no_bc.dot(primal_solutions[i])
    for j in range(n_space_dofs):
        dual_rhs[j] = last_dual_solution[j + n_space_dofs]

    dual_solutions.append(
        scipy.sparse.linalg.spsolve(dual_matrix, dual_rhs))

    last_dual_solution = dual_solutions[-1]
end_execution = time.time()
execution_time_FOM = end_execution - start_execution
print("Dual FOM time:   " + str(execution_time_FOM))

dual_solutions = dual_solutions[::-1]

for i, dual_solution in enumerate(dual_solutions):
	save_vtk(OUTPUT_PATH + cycle + f"/py_dual_solution{i:05}.vtk", {"displacement": dof_matrix.dot(dual_solution[0:n_dofs["space"]]), "velocity": dof_matrix.dot(dual_solution[n_dofs["space"]:2 * n_dofs["space"]])}, grid, cycle=i, time=list_coordinates_t[i][0])

print("\n\nTO DO @Hendrik")
