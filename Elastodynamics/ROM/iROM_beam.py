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
from iPOD import iPOD, ROM_update, ROM_update_dual, reduce_matrix, reduce_vector, project_vector
#import imageio

PLOTTING = False
MOTHER_PATH = "/home/hendrik/Code/MORe_DWR/Elastodynamics/"
OUTPUT_PATH = MOTHER_PATH + "/Data/3D/Rod/"
cycle = "cycle=1"
SAVE_PATH = MOTHER_PATH + "Data/ROM/" + cycle + "/"
# SAVE_PATH = cycle + "/output_ROM/"

#"../../FOM/slabwise/output_" + CASE + "/dim=1/"

ENERGY_PRIMAL_DISPLACEMENT = 0.99999
ENERGY_PRIMAL_VELOCITY = 0.999999
ENERGY_DUAL = 0.999999

CG_ORDER = 1

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
if not os.path.exists(SAVE_PATH + "movie/"):    
    os.makedirs(SAVE_PATH + "movie/")

# %% vtk plotting requeiremnets


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
            lines.append(str((time)))

    lines += grid

    for key, value in solution.items():
        lines.append(f"SCALARS {key}_x double 1")
        lines.append("LOOKUP_TABLE default")
        lines.append(" ".join(np.round(value[0::3], decimals=7).astype(
            np.double).astype(str)) + " ")
        
        lines.append(f"SCALARS {key}_y double 1")
        lines.append("LOOKUP_TABLE default")
        lines.append(" ".join(np.round(value[1::3], decimals=7).astype(
            np.double).astype(str)) + " ")
        
        lines.append(f"SCALARS {key}_z double 1")
        lines.append("LOOKUP_TABLE default")
        lines.append(" ".join(np.round(value[2::3], decimals=7).astype(
            np.double).astype(str)) + " ")


    with open(file_name, "w") as file:
        file.write("\n".join(lines))


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
# print(grid)
dof_vector = np.loadtxt(OUTPUT_PATH + cycle + "/dof.txt").astype(int)
dof_matrix = scipy.sparse.dok_matrix((dof_vector.shape[0], dof_vector.max()+1))
for i, j in enumerate(list(dof_vector)):
    dof_matrix[i, j] = 1.


print(f"\n{'-'*12}\n| {cycle}: |\n{'-'*12}\n")
# NO BC
[data, row, column] = np.loadtxt(OUTPUT_PATH + cycle + "/matrix_no_bc.txt")
matrix_no_bc = scipy.sparse.csr_matrix(
    (data, (row.astype(int), column.astype(int))))

[data, row, column] = np.loadtxt(
    OUTPUT_PATH + cycle + "/dual_matrix_no_bc.txt")
functional_matrix_no_bc = scipy.sparse.csr_matrix(
    (data, (row.astype(int), column.astype(int))))


rhs_no_bc = []
for f in sorted([f for f in os.listdir(OUTPUT_PATH + cycle)
                if "dual" not in f and "rhs_no_bc" in f]):
    rhs_no_bc.append(np.loadtxt(OUTPUT_PATH + cycle + "/" + f))

boundary_ids = np.loadtxt(OUTPUT_PATH + cycle +
                          "/boundary_id.txt").astype(int)

# %% applying BC to primal matrix
primal_matrix = matrix_no_bc.tocsr()
primal_system_rhs  = []
for row in boundary_ids:
    for col in primal_matrix.getrow(row).nonzero()[1]:
        primal_matrix[row, col] = 1. if row == col else 0.
    
    for rhs_no_bc_sample in rhs_no_bc:
        primal_system_rhs.append(rhs_no_bc_sample)
        primal_system_rhs[-1][row] = 0.0
        # for in_bc in range(len(rhs_no_bc)):
        #     if row == col: 
        #         rhs_no_bc[in_bc][col] = 1. 

"""
dual_rhs_no_bc = []
for f in sorted([f for f in os.listdir(
        OUTPUT_PATH + cycle) if "dual_rhs_no_bc" in f]):
    dual_rhs_no_bc.append(np.loadtxt(OUTPUT_PATH + cycle + "/" + f))
"""


initial_solution = np.loadtxt(OUTPUT_PATH + cycle + "/initial_solution.txt")

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
# n_dofs["space"] per unknown u and v -> 2*n_dofs["space"] for one block
n_dofs = {"space": coordinates_x.shape[1], "time": coordinates_t.shape[0]}


cfl = (list_coordinates_t[0][1] - list_coordinates_t[0][0]) / \
    (np.sqrt(2)*(coordinates_x[0][1] - coordinates_x[0][0]))


# ------------
# %% reduced linear equation system size, since solution of first time DoF can be enforced
# -----------
#  A  |  B
# -----------
#  C  |  D
# -----------
# need this for   dual problem
A = matrix_no_bc[:2*n_dofs["space"], :2*n_dofs["space"]]
# need this for   dual problem
B = matrix_no_bc[:2*n_dofs["space"], 2*n_dofs["space"]:]
# need this for primal and dual problem
C = matrix_no_bc[2*n_dofs["space"]:, :2*n_dofs["space"]]
# need this for primal and dual problem
D = matrix_no_bc[2*n_dofs["space"]:, 2*n_dofs["space"]:]


A_wbc = primal_matrix[:2*n_dofs["space"], :2*n_dofs["space"]]
# need this for   dual problem
B_wbc = primal_matrix[:2*n_dofs["space"], 2*n_dofs["space"]:]
# need this for primal and dual problem
C_wbc = primal_matrix[2*n_dofs["space"]:, :2*n_dofs["space"]]
# need this for primal and dual problem
D_wbc = primal_matrix[2*n_dofs["space"]:, 2*n_dofs["space"]:]

print(f"C.shape = {C.shape}")
print(f"D.shape = {D.shape}")
print(f"matrix_no_bc.shape = {matrix_no_bc.shape}")

# ------------
# %% primal FOM solve
# solve D x_2 = b_2 - C x_0 since x_1 = x_0 due to continuity
start_execution = time.time()
last_primal_solution = np.zeros((2*n_dofs["space"],))
last_primal_solution[:] = initial_solution[:]
primal_solutions = [initial_solution[:]]
for i in range(n_slabs):
    # creating primal rhs and applying BC to it
    primal_rhs = primal_system_rhs[i][2*n_dofs["space"]:].copy() - C_wbc.dot(last_primal_solution)

    primal_solutions.append(
        scipy.sparse.linalg.spsolve(D_wbc, primal_rhs))
    last_primal_solution = primal_solutions[-1]
end_execution = time.time()
execution_time_FOM = end_execution - start_execution

print("Primal FOM time:   " + str(execution_time_FOM))
print("CFL number:        " +
      str(cfl))
print("n_dofs[space] =", n_dofs["space"])

# save_vtk(OUTPUT_PATH + cycle + "/py_solution00000.vtk", {"displacement": dof_matrix.dot(
#     initial_solution[0:n_dofs["space"]]), "velocity": dof_matrix.dot(initial_solution[n_dofs["space"]:2 * n_dofs["space"]])}, grid, cycle=0, time=0.)
for i, primal_solution in enumerate(primal_solutions):
    save_vtk(OUTPUT_PATH + cycle + f"/py_solution{i:05}.vtk", {"displacement": dof_matrix.dot(primal_solution[0:n_dofs["space"]]), "velocity": dof_matrix.dot(
        primal_solution[n_dofs["space"]:2 * n_dofs["space"]])}, grid, cycle=i, time=(list_coordinates_t[i-1][1] if i > 0 else 0.))

# %% applying BC to dual matrix
# dual_matrix = matrix_no_bc.T.tocsr()
# for row in range(2*n_dofs["space"]):
#     for col in dual_matrix.getrow(row).nonzero()[1]:
#         dual_matrix[row, col] = 1. if row == col else 0.

# ------------
# %% reduced linear equation system size, since solution of first time DoF can be enforced
J_1 = functional_matrix_no_bc[:2*n_dofs["space"], :2*n_dofs["space"]]
J_2 = functional_matrix_no_bc[:2*n_dofs["space"], 2*n_dofs["space"]:]
J_3 = functional_matrix_no_bc[2*n_dofs["space"]:, :2*n_dofs["space"]]
J_4 = functional_matrix_no_bc[2*n_dofs["space"]:, 2*n_dofs["space"]:]

print(f"J_1.shape = {J_1.shape}")
print(f"J_2.shape = {J_2.shape}")

# ------------
# %% dual FOM solve
start_execution = time.time()
last_dual_solution = np.zeros((2*n_dofs["space"],))
# last_dual_solution[:] = initial_solution[:]
# zero initial condition for dual problem
dual_solutions = [np.zeros((2*n_dofs["space"],))]
for i in list(range(n_slabs))[::-1]:
    # creating dual rhs and applying BC to it
    # functional_matrix_no_bc.dot(primal_solutions[i])
    dual_rhs = J_2.dot(primal_solutions[i])
    if i > 0:
        dual_rhs += J_1.dot(primal_solutions[i-1])
    else:
        dual_rhs += J_1.dot(initial_solution)
    dual_rhs -= B.T.dot(last_dual_solution)
    # dual_rhs = -1. * B.T.dot(last_dual_solution)

    dual_solutions.append(
        scipy.sparse.linalg.spsolve(A.T, dual_rhs))

    last_dual_solution = dual_solutions[-1]
end_execution = time.time()
execution_time_FOM = end_execution - start_execution
print("Dual FOM time:   " + str(execution_time_FOM))

dual_solutions = dual_solutions[::-1]

for i, dual_solution in enumerate(dual_solutions):
    save_vtk(OUTPUT_PATH + cycle + f"/py_dual_solution{i:05}.vtk", {"displacement": dof_matrix.dot(dual_solution[0:n_dofs["space"]]), "velocity": dof_matrix.dot(
        dual_solution[n_dofs["space"]:2 * n_dofs["space"]])}, grid, cycle=i, time=(list_coordinates_t[i-1][1] if i > 0 else 0.))
# for i, primal_solution in enumerate(primal_solutions):
#     save_vtk(OUTPUT_PATH + cycle + f"/py_solution{i:05}.vtk", {"displacement": dof_matrix.dot(primal_solution[0:n_dofs["space"]]), "velocity": dof_matrix.dot(
#         primal_solution[n_dofs["space"]:2 * n_dofs["space"]])}, grid, cycle=i, time=(list_coordinates_t[i-1][1] if i > 0 else 0.))

# %% goal functionals
J = {"u_h": 0., "u_r": 0.}
J_h_t = np.empty([n_slabs, 1])


for i in range(1,n_slabs+1):
    u_2 = primal_solutions[i][:]
    if i > 0:
        u_1 = primal_solutions[i-1][:]
    else:
        u_1 = initial_solution[:]
    J_h_t[i-1] = u_1.dot(J_1.dot(u_1)+J_2.dot(u_2)) + \
        u_2.dot(J_3.dot(u_1)+J_4.dot(u_2))
J["u_h"] = np.sum(J_h_t)


# %%
# check reducablility

bunch_size = 1 #len(primal_solutions)  # 1

total_energy = {"displacement": 0, "velocity": 0}
pod_basis = {"displacement": np.empty([0, 0]), "velocity": np.empty([0, 0])}
bunch = {"displacement": np.empty([0, 0]), "velocity": np.empty([0, 0])}
singular_values = {"displacement": np.empty(
    [0, 0]), "velocity": np.empty([0, 0])}

total_energy_dual = {"displacement": 0, "velocity": 0}
pod_basis_dual = {"displacement": np.empty(
    [0, 0]), "velocity": np.empty([0, 0])}
bunch_dual = {"displacement": np.empty([0, 0]), "velocity": np.empty([0, 0])}
singular_values_dual = {"displacement": np.empty(
    [0, 0]), "velocity": np.empty([0, 0])}

# total_energy_displacement = 0
# pod_basis_displacement = np.empty([0, 0])
# bunch_displacement = np.empty([0, 0])
# singular_values_displacement = np.empty([0, 0])

# total_energy_displacement = 0
# pod_basis_displacement = np.empty([0, 0])
# bunch_displacement = np.empty([0, 0])
# singular_values_displacement = np.empty([0, 0])

for primal_solution in primal_solutions[0:1]:
    pod_basis["displacement"], bunch["displacement"], singular_values["displacement"], total_energy["displacement"] \
        = iPOD(pod_basis["displacement"],
               bunch["displacement"],
               singular_values["displacement"],
               primal_solution[0:n_dofs["space"]],
               total_energy["displacement"],
               ENERGY_PRIMAL_DISPLACEMENT,
               bunch_size)      
    pod_basis["velocity"], bunch["velocity"], singular_values["velocity"], total_energy["velocity"] \
        = iPOD(pod_basis["velocity"],
               bunch["velocity"],
               singular_values["velocity"],
               primal_solution[n_dofs["space"]:2 * n_dofs["space"]],
               total_energy["velocity"],
               ENERGY_PRIMAL_VELOCITY,
               bunch_size)
        
for dual_solution in dual_solutions[0:1]:
    pod_basis_dual["displacement"], bunch_dual["displacement"], singular_values_dual["displacement"], total_energy_dual["displacement"] \
        = iPOD(pod_basis_dual["displacement"],
               bunch_dual["displacement"],
               singular_values_dual["displacement"],
               dual_solution[0:n_dofs["space"]],
               total_energy_dual["displacement"],
               ENERGY_DUAL,
               bunch_size)
    pod_basis_dual["velocity"], bunch_dual["velocity"], singular_values_dual["velocity"], total_energy_dual["velocity"] \
        = iPOD(pod_basis_dual["velocity"],
               bunch_dual["velocity"],
               singular_values_dual["velocity"],
               dual_solution[n_dofs["space"]:2 * n_dofs["space"]],
               total_energy_dual["velocity"],
               ENERGY_DUAL,
               bunch_size)
    

print(pod_basis["displacement"].shape[1])
print(pod_basis["velocity"].shape[1])
print(pod_basis_dual["displacement"].shape[1])
print(pod_basis_dual["velocity"].shape[1])

# compute reduced matrices
# needed for dual
A_reduced = reduce_matrix(A, pod_basis_dual, pod_basis_dual)
B_reduced = reduce_matrix(B, pod_basis_dual, pod_basis_dual)
J_1_reduced = reduce_matrix(J_1, pod_basis_dual, pod_basis)
J_2_reduced = reduce_matrix(J_1, pod_basis_dual, pod_basis)
# needed for primal
C_reduced = reduce_matrix(C, pod_basis, pod_basis)
D_reduced = reduce_matrix(D, pod_basis, pod_basis)

# %% primal ROM solve
reduced_solutions = []
reduced_solution_old = reduce_vector(initial_solution[:], pod_basis)

reduced_dual_solutions = []
reduced_dual_solution_old = reduce_vector(dual_solutions[0], pod_basis_dual)
forward_reduced_dual_solution = np.zeros_like(reduce_vector(dual_solutions[0], pod_basis_dual))

projected_reduced_solutions = [project_vector(reduce_vector(initial_solution[:], pod_basis), pod_basis)]
projected_reduced_solutions_before_enrichment = []

projected_reduced_dual_solutions = [project_vector(reduce_vector(dual_solutions[0], pod_basis_dual), pod_basis_dual)]

dual_residual = []
dual_residual.append(0)

temporal_interval_error = []
temporal_interval_error_incidactor = []

tol = 5e-4/(n_slabs)
tol_rel = 2e-2
tol_dual = 5e-1
forwardsteps = 5

# print("tol =     " + str(tol))
print("tol_rel       = " + str(tol_rel))
print("tol           = " + str(tol))
print(f"forward steps = {forwardsteps}")
print(" ")
start_execution = time.time()
extime_solve = 0.0
extime_dual_solve = 0.0
extime_error = 0.0
extime_update = 0.0

for i in range(n_slabs):
    start_time = time.time()
    # primal ROM solve
    reduced_rhs = reduce_vector(
        rhs_no_bc[i][2*n_dofs["space"]:].copy(), pod_basis) - C_reduced.dot(reduced_solution_old)
    reduced_solution = np.linalg.solve(D_reduced, reduced_rhs)

    projected_reduced_solutions.append(project_vector(reduced_solution, pod_basis))
    
    # reduced_dual_rhs=J_2_reduced.dot(reduced_solution) + J_1_reduced.dot(reduced_solution_old)
    # reduced_dual_rhs -= B_reduced.T.dot(forward_reduced_dual_solutions[-1])

    # reduced_dual_solution=np.linalg.solve(A_reduced.T, reduced_dual_rhs)


    # ATTENTION WE HAVE TO SPLIT THE REDUCED SOLUTION SINCE IT CONTAINS NOW MORE THEN ONE TIMESTEP


    reduced_solution_old = reduced_solution

    # print(np.linalg.norm(projected_reduced_solutions[i+1][:n_dofs["space"]]-primal_solutions[i+1]
    #                      [:n_dofs["space"]])/np.linalg.norm(primal_solutions[i+1][:n_dofs["space"]]))
    # print(np.linalg.norm(projected_reduced_solutions[i+1][n_dofs["space"]:]-primal_solutions[i+1]
    #                      [n_dofs["space"]:])/np.linalg.norm(primal_solutions[i+1][n_dofs["space"]:]))
    # if np.linalg.norm(dual_solutions[i+1][n_dofs["space"]:]) != 0.0 and np.linalg.norm(dual_solutions[i+1][:n_dofs["space"]]) != 0.0:
    #     print(np.linalg.norm(projected_reduced_dual_solutions[i+1][:n_dofs["space"]]-dual_solutions[i+1]
    #                          [:n_dofs["space"]])/np.linalg.norm(dual_solutions[i+1][:n_dofs["space"]]))
    #     print(np.linalg.norm(projected_reduced_dual_solutions[i+1][n_dofs["space"]:]-dual_solutions[i+1]
    #                          [n_dofs["space"]:])/np.linalg.norm(dual_solutions[i+1][n_dofs["space"]:]))
    # print(" ")


# %% postprocessing
J_r_t = np.empty([n_slabs, 1])
for i in range(1,n_slabs+1):
    u_2 = projected_reduced_solutions[i][:]
    u_1 = projected_reduced_solutions[i-1][:]
    J_r_t[i-1] = u_1.dot(J_1.dot(u_1)+J_2.dot(u_2)) + \
        u_2.dot(J_3.dot(u_1)+J_4.dot(u_2))
J["u_r"] = np.sum(J_r_t)



end_execution = time.time()
execution_time_ROM = end_execution - start_execution
print("ROM time:        " + str(execution_time_ROM))
print("CFL number:        " + str(cfl))

for i, projected_reduced_solution in enumerate(projected_reduced_solutions):
    save_vtk(OUTPUT_PATH + cycle + f"/primal_solution{i:05}.vtk",
             {"displacement": dof_matrix.dot(projected_reduced_solution[0:n_dofs["space"]]), "velocity":    dof_matrix.dot(projected_reduced_solution[n_dofs["space"]:2 * n_dofs["space"]])}, grid, cycle=i, time=(list_coordinates_t[i-1][1] if i > 0 else 0.))

for i, projected_reduced_dual_solution in enumerate(projected_reduced_dual_solutions):
    save_vtk(OUTPUT_PATH + cycle + f"/dual_solution{i:05}.vtk",
             {"displacement": dof_matrix.dot(projected_reduced_dual_solution[0:n_dofs["space"]]), "velocity":    dof_matrix.dot(projected_reduced_dual_solution[n_dofs["space"]:2 * n_dofs["space"]])}, grid, cycle=i, time=(list_coordinates_t[i-1][1] if i > 0 else 0.))

    # save_vtk(OUTPUT_PATH + cycle + f"/py_solution{i:05}.vtk",
# 	save_vtk(OUTPUT_PATH + cycle + f"/py_solution{i+1:05}.vtk", {"displacement": dof_matrix.dot(primal_solution[0:n_dofs["space"]]), \
#             "velocity": dof_matrix.dot(primal_solution[n_dofs["space"]:2 * n_dofs["space"]])}, grid, cycle=i+1, time=list_coordinates_t[i][1])
error_primal_displacement = []
error_dual_displacement = []
error_primal_velo = []
error_dual_velo = []
for i in range(n_slabs+1):
    if np.linalg.norm(primal_solutions[i][n_dofs["space"]:]) != 0.0 and np.linalg.norm(primal_solutions[i][:n_dofs["space"]]) != 0.0:
        error_primal_displacement.append(np.linalg.norm(primal_solutions[i][:n_dofs["space"]] - projected_reduced_solutions[i][:n_dofs["space"]])/ np.linalg.norm(primal_solutions[i][:n_dofs["space"]]) )
        error_primal_velo.append(np.linalg.norm(primal_solutions[i][n_dofs["space"]:] - projected_reduced_solutions[i][n_dofs["space"]:])/ np.linalg.norm(primal_solutions[i][n_dofs["space"]:]) )
    else:
        error_primal_displacement.append(0)
        error_primal_velo.append(0)
    if np.linalg.norm(dual_solutions[i][n_dofs["space"]:]) != 0.0 and np.linalg.norm(dual_solutions[i][:n_dofs["space"]]) != 0.0:
        error_dual_displacement.append(np.linalg.norm(dual_solutions[i][:n_dofs["space"]] - projected_reduced_dual_solutions[i][:n_dofs["space"]])/ np.linalg.norm(projected_reduced_dual_solutions[i][:n_dofs["space"]]) )
        error_dual_velo.append(np.linalg.norm(dual_solutions[i][n_dofs["space"]:] - projected_reduced_dual_solutions[i][n_dofs["space"]:])/np.linalg.norm(dual_solutions[i][n_dofs["space"]:]) )
    else:
        error_dual_displacement.append(0)
        error_dual_velo.append(0)
# %%


time_step_size = 40.0 / (n_dofs["time"] / 2)

# plot sigs
plt.rc('text', usetex=True)
# plt.rcParams["figure.figsize"] = (10,2)
plt.plot(np.arange(0, pod_basis["displacement"].shape[1]),
         singular_values["displacement"], label="displacement")
plt.plot(np.arange(0, pod_basis["velocity"].shape[1]),
         singular_values["velocity"], label="velocity")
plt.plot(np.arange(0, pod_basis_dual["displacement"].shape[1]),
         singular_values_dual["displacement"], label="displacement_dual")
plt.plot(np.arange(0, pod_basis_dual["velocity"].shape[1]),
         singular_values_dual["velocity"], label="velocity_dual")
plt.grid()
plt.yscale('log')
plt.legend()
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.show()


# plot norm of displace,mnet
plt.rc('text', usetex=True)
# plt.rcParams["figure.figsize"] = (10,2)
plt.plot(np.arange(0, n_slabs+1),
         error_primal_displacement, label="primal")
plt.plot(np.arange(0, n_slabs+1),
         error_dual_displacement, label="dual")
# plt.yscale('log')
plt.legend()
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.show()

# plot norm of velocity
plt.rc('text', usetex=True)
# plt.rcParams["figure.figsize"] = (10,2)
plt.plot(np.arange(0, n_slabs+1),
         error_primal_velo, label="primal")
plt.plot(np.arange(0, n_slabs+1),
         error_dual_velo, label="dual")
# plt.yscale('log')
plt.legend()
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.show()


# plot temporal evolution of cost funtiponal
plt.rc('text', usetex=True)
# plt.rcParams["figure.figsize"] = (10,2)
plt.plot(np.arange(0, n_slabs*time_step_size, time_step_size),
          J_h_t, color='r', label="$u_h$")
plt.plot(np.arange(0, n_slabs*time_step_size, time_step_size),
          J_r_t, '--', c='#1f77b4', label="$u_N$")
plt.grid()
plt.legend()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.xlabel('$t \; [$s$]$')
plt.ylabel("$J(u)\\raisebox{-.5ex}{$|$}_{Q_l}$")
plt.xlim([0, n_slabs*time_step_size])
#plt.title("temporal evaluation of cost funtional")

plt.show()


# plot temporal evolution of cost funtiponal
plt.rcParams["figure.figsize"] = (10,2)
# plt.plot(np.arange(0, n_slabs*time_step_size, time_step_size),
#          temporal_interval_error, c='#1f77b4', label="estimate")
plt.plot(np.arange(0, n_slabs*time_step_size, time_step_size),
         J_h_t-J_r_t, color='r', label="error")
plt.grid()
plt.legend()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.xlabel('$t \; [$s$]$')
plt.ylabel("$error$")
plt.xlim([0, n_slabs*time_step_size])

plt.show()