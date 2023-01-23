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
from auxiliaries import save_vtk, read_in_LES, apply_boundary_conditions, read_in_discretization,solve_primal_FOM_step, solve_dual_FOM_step, solve_primal_ROM_step, reorder_matrix,reorder_vector,error_estimator,save_solution_txt, load_solution_txt, evaluate_cost_functional
#import imageio

PLOTTING = False
MOTHER_PATH = "/home/hendrik/Code/MORe_DWR/Elastodynamics/"
OUTPUT_PATH = MOTHER_PATH + "/Data/3D/Rod/"
OUTPUT_PATH_DUAL = MOTHER_PATH + "Dual_Elastodynamics/Data/3D/Rod/"
cycle = "cycle=1"
SAVE_PATH = MOTHER_PATH + "Data/ROM/" + cycle + "/"

LOAD_SOLUTION = True

print(f"\n{'-'*12}\n| {cycle}: |\n{'-'*12}\n")

ENERGY_DUAL = 0.999999
ENERGY_PRIMAL = {"displacement": 0.99999999, \
                 "velocity":     0.99999999}

# %% read in properties connected to discretization
n_dofs, slab_properties, index2measuredisp, dof_matrix, grid = read_in_discretization(OUTPUT_PATH + cycle)

# %% Reading in matricies and rhs without bc
matrix_no_bc, rhs_no_bc = read_in_LES(OUTPUT_PATH + cycle, "/matrix_no_bc.txt", "primal_rhs_no_bc")
dual_matrix_no_bc, dual_rhs_no_bc = read_in_LES(OUTPUT_PATH_DUAL + cycle, "/dual_matrix_no_bc.txt", "dual_rhs_no_bc")
# %% Enforcing BC to primal und dual systems 
primal_matrix, primal_system_rhs = apply_boundary_conditions(matrix_no_bc, rhs_no_bc, OUTPUT_PATH + cycle + "/boundary_id.txt")
dual_matrix, dual_system_rhs = apply_boundary_conditions(dual_matrix_no_bc, dual_rhs_no_bc, OUTPUT_PATH + cycle + "/boundary_id.txt")

# %% read in IC
initial_solution = np.loadtxt(OUTPUT_PATH + cycle + "/initial_solution.txt")

# %% Reorder matrices and vectors
# reorder matricies
matrix_no_bc = reorder_matrix(matrix_no_bc, slab_properties, n_dofs)
dual_matrix_no_bc = reorder_matrix(dual_matrix_no_bc, slab_properties, n_dofs)
primal_matrix = reorder_matrix(primal_matrix, slab_properties, n_dofs)
dual_matrix = reorder_matrix(dual_matrix, slab_properties, n_dofs)


# reorder vectors
for j in range(slab_properties["n_total"]):
    rhs_no_bc[j] = reorder_vector(rhs_no_bc[j], slab_properties, n_dofs)
    dual_rhs_no_bc[j] = reorder_vector(dual_rhs_no_bc[j], slab_properties, n_dofs)
    primal_system_rhs[j] = reorder_vector(primal_system_rhs[j], slab_properties, n_dofs)
    dual_system_rhs[j] = reorder_vector(dual_system_rhs[j], slab_properties, n_dofs)
        
# %% Definition of submatricies

# PRIMAL 
# --------------    
#  ~  |    ~        x_1
# --------------
#     |             x_2
#  C  |    D        ...
#     |             x_n
# --------------

# primal problem matricies
C = matrix_no_bc[n_dofs["time_step"]:, :n_dofs["time_step"]]
D = matrix_no_bc[n_dofs["time_step"]:, n_dofs["time_step"]:]

C_wbc = primal_matrix[n_dofs["time_step"]:, :n_dofs["time_step"]]
D_wbc = primal_matrix[n_dofs["time_step"]:, n_dofs["time_step"]:]


# Dual 
# --------------
#        |           z_1
#    A   |  B        ...
#        |           z_n-1
# -------------- 
#    ~   |  ~        z_n
# --------------

# dual problem matricies
A = dual_matrix_no_bc[:-n_dofs["time_step"], :-n_dofs["time_step"]]
B = dual_matrix_no_bc[:-n_dofs["time_step"], -n_dofs["time_step"]:]

A_wbc = dual_matrix[:-n_dofs["time_step"], :-n_dofs["time_step"]]
B_wbc = dual_matrix[:-n_dofs["time_step"], -n_dofs["time_step"]:]

print(f"A.shape = {A.shape}")
print(f"B.shape = {B.shape}")
print(f"C.shape = {C.shape}")
print(f"D.shape = {D.shape}")
print(f"matrix_no_bc.shape = {matrix_no_bc.shape}")

# ------------
# %% Primal FOM solve
start_execution = time.time()


if not LOAD_SOLUTION:
    primal_solutions = {"value": [initial_solution], "time": [0.]}

    for i in range(slab_properties["n_total"]):
        primal_solutions = solve_primal_FOM_step(primal_solutions, D_wbc, C_wbc, primal_system_rhs[i], slab_properties, n_dofs, i)

else:
    primal_solutions = load_solution_txt(SAVE_PATH + "/py_solution")

end_execution = time.time()

execution_time_FOM = end_execution - start_execution

print("Primal FOM time:   " + str(execution_time_FOM))
print("n_dofs[space] =", n_dofs["space"])


save_solution_txt(SAVE_PATH + "/py_solution", primal_solutions)

# for i, primal_solution in enumerate(primal_solutions["value"]):
#     save_solution_txt(SAVE_PATH + "/py_solution", primal_solution,i)
# np.savetxt(SAVE_PATH + "/py_solution_time.txt", primal_solutions["time"])   


for i, primal_solution in enumerate(primal_solutions["value"]):
    save_vtk(SAVE_PATH + f"/py_solution{i:05}.vtk", {"displacement": dof_matrix.dot(primal_solution[0:n_dofs["space"]]), "velocity": dof_matrix.dot(
        primal_solution[n_dofs["space"]:2 * n_dofs["space"]])}, grid, cycle=i, time=primal_solutions["time"][i])

primal_solutions_slab =  {"value": [], "time": []}
for i in range(slab_properties["n_total"]):
    primal_solutions_slab["value"].append(np.hstack( primal_solutions["value"][i*(n_dofs['solperstep']): (i+1)*(n_dofs['solperstep'])+1]).T)
    primal_solutions_slab["time"].append(slab_properties["time_points"][i])


# %% dual FOM solve
start_execution = time.time()
last_dual_solution = np.zeros((n_dofs["time_step"],))


# for debugging:
# dual_system_rhs = primal_system_rhs[::-1] 
if not LOAD_SOLUTION:
    dual_solutions = {"value": [last_dual_solution], "time": [slab_properties["time_points"][-1][-1]]}

    for i in list(range(slab_properties["n_total"]))[::-1]:   
        dual_solutions = solve_dual_FOM_step(dual_solutions, A_wbc, B_wbc, dual_system_rhs[i], slab_properties, n_dofs, i)

        
    dual_solutions["value"] = dual_solutions["value"][::-1]
    dual_solutions["time"] = dual_solutions["time"][::-1]

else:
    dual_solutions = {"value": [], "time": []}
    i = 0
    for f in sorted([f for f in os.listdir(OUTPUT_PATH_DUAL + cycle) if "dual_solution" in f]):
        tmp_sol = np.loadtxt(OUTPUT_PATH_DUAL + cycle + "/" + f)
        for j in range(slab_properties["n_time_unknowns"]):
            dual_solutions["value"].append(tmp_sol[j*n_dofs["time_step"]:(j+1)*n_dofs["time_step"]])
            dual_solutions["time"].append(slab_properties["time_points"][i][j])
        i += 1
        # for j in slab_properties["ordering"][slab_properties["ordering"] < n_dofs["solperstep"]]:
        #     dual_solutions.append(tmp_sol[j*n_dofs["time_step"]:(j+1)*n_dofs["time_step"]])
    # final condition = 0
    dual_solutions["value"].append(np.zeros((n_dofs["time_step"],)))
    dual_solutions["time"].append(slab_properties["time_points"][-1][-1])
    # dual_solutions.append(np.zeros((n_dofs["time_step"],)))
        
end_execution = time.time()
execution_time_FOM = end_execution - start_execution
print("Dual FOM time:   " + str(execution_time_FOM))


for i, dual_solution in enumerate(dual_solutions["value"]):
    save_vtk(SAVE_PATH + f"/py_dual_solution{i:05}.vtk", {"displacement": dof_matrix.dot(dual_solution[0:n_dofs["space"]]), "velocity": dof_matrix.dot(
        dual_solution[n_dofs["space"]:2 * n_dofs["space"]])}, grid, cycle=i, time=dual_solutions["time"][i])


dual_solutions_slab =  {"value": [], "time": []}
for i in range(slab_properties["n_total"]):
    dual_solutions_slab["value"].append(np.hstack( dual_solutions["value"][i*(n_dofs['solperstep']): (i+1)*(n_dofs['solperstep'])+1]).T)
    dual_solutions_slab["time"].append(slab_properties["time_points"][i])


# %% Definition and initialization of ROM ingredients

bunch_size = 1  # len(primal_solutions)  # 1

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


for primal_solution in primal_solutions["value"][0:2]:
    pod_basis["displacement"], bunch["displacement"], singular_values["displacement"], total_energy["displacement"] \
        = iPOD(pod_basis["displacement"],
               bunch["displacement"],
               singular_values["displacement"],
               primal_solution[0:n_dofs["space"]],
               total_energy["displacement"],
               ENERGY_PRIMAL["displacement"],
               bunch_size)
    pod_basis["velocity"], bunch["velocity"], singular_values["velocity"], total_energy["velocity"] \
        = iPOD(pod_basis["velocity"],
               bunch["velocity"],
               singular_values["velocity"],
               primal_solution[n_dofs["space"]:2 * n_dofs["space"]],
               total_energy["velocity"],
               ENERGY_PRIMAL["velocity"],
               bunch_size)

for dual_solution in dual_solutions["value"]:#[0:1]:
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
C_reduced = reduce_matrix(C, pod_basis, pod_basis)
D_reduced = reduce_matrix(D, pod_basis, pod_basis)

# %% Primal ROM solve
# reduced_solutions = {"value": [], "time": []}

reduced_solutions = []
reduced_solution_old = reduce_vector(initial_solution[:], pod_basis)

projected_reduced_solutions = {"value": [project_vector(reduce_vector(initial_solution[:], pod_basis), pod_basis)], 
                               "time": [0.]}

last_projected_reduced_solution = projected_reduced_solutions
projected_reduced_solutions_before_enrichment = []


temporal_interval_error = []
temporal_interval_error_shorti = []
temporal_interval_error_incidactor = []
temporal_interval_error_relative = []

tol = 5e-4/(slab_properties["n_total"])
tol_rel = 1e-2
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

n_dofs["reduced_primal"] = pod_basis["displacement"].shape[1] + pod_basis["velocity"].shape[1]
for i in range(slab_properties["n_total"]):

    start_time = time.time()
    # primal ROM solve
    projected_reduced_solutions, reduced_solution_old = solve_primal_ROM_step(projected_reduced_solutions, reduced_solution_old, D_reduced, C_reduced, rhs_no_bc[i], pod_basis, slab_properties, n_dofs, i)

    temporal_interval_error.append(error_estimator(projected_reduced_solutions, dual_solutions, matrix_no_bc, rhs_no_bc[i].copy(), slab_properties))

    temporal_interval_error_relative.append(temporal_interval_error[-1] / \
                            np.abs(temporal_interval_error[-1] + evaluate_cost_functional(projected_reduced_solutions,dual_rhs_no_bc[i].copy(), slab_properties,i)))

    if temporal_interval_error_relative[-1] > tol_rel:
        print(f"{i}: {slab_properties['n_time_unknowns']*i} - {slab_properties['n_time_unknowns']*(i+1)}")
        pod_basis, C_reduced, D_reduced, projected_reduced_solutions["value"][-slab_properties["n_time_unknowns"]:], singular_values, total_energy = ROM_update( 
                    pod_basis, 
                    projected_reduced_solutions["value"][-slab_properties["n_time_unknowns"]-1], # last solution of slab before
                    C_wbc, 
                    D_wbc, 
                    C, 
                    D, 
                    rhs_no_bc[i][n_dofs["time_step"]:].copy(), 
                    slab_properties["n_time_unknowns"], 
                    singular_values, 
                    total_energy, 
                    ENERGY_PRIMAL,
                    slab_properties["n_time_unknowns"], 
                    n_dofs)


        n_dofs["reduced_primal"] = pod_basis["displacement"].shape[1] + pod_basis["velocity"].shape[1]
        reduced_solution_old = reduce_vector(projected_reduced_solutions["value"][-1],pod_basis)
    # last_projected_reduced_solution = projected_reduced_solutions["value"][-1]

end_execution = time.time()
execution_time_ROM = end_execution - start_execution
print("ROM time:        " + str(execution_time_ROM))

# save projected reduced solutions to vtk
for i, projected_reduced_solution in enumerate(projected_reduced_solutions["value"]):
    save_vtk(SAVE_PATH + f"/projected_solution{i:05}.vtk", {"displacement": dof_matrix.dot(projected_reduced_solution[0:n_dofs["space"]]), "velocity": dof_matrix.dot(
        projected_reduced_solution[n_dofs["space"]:2 * n_dofs["space"]])}, grid, cycle=i, time=projected_reduced_solutions["time"][i])

# generate slabs for projected reduced solutions
projected_reduced_solutions_slab =  {"value": [], "time": []}
for i in range(slab_properties["n_total"]):
    projected_reduced_solutions_slab["value"].append(np.hstack( projected_reduced_solutions["value"][i*(n_dofs['solperstep']): (i+1)*(n_dofs['solperstep'])+1]).T)
    projected_reduced_solutions_slab["time"].append(slab_properties["time_points"][i])

# %% Computing reduced and full Cost functional

J_r_t = np.empty([slab_properties["n_total"], 1])
J_h_t = np.empty([slab_properties["n_total"], 1])   
# TODO: is this correct? Yes, julian it is correct ;)
for i in range(slab_properties["n_total"]):
    J_r_t[i] = projected_reduced_solutions_slab["value"][i].dot(dual_rhs_no_bc[i])
    J_h_t[i] = primal_solutions_slab["value"][i].dot(dual_rhs_no_bc[i])


# %% Error evaluations for primal and dual solutions

error_primal_displacement = []
error_primal_velo = []
error_primal_displacement_pointwise = []
for i in range(slab_properties["n_total"]+1):
    if np.linalg.norm(primal_solutions["value"][i][n_dofs["space"]:]) != 0.0 and np.linalg.norm(primal_solutions["value"][i][:n_dofs["space"]]) != 0.0:
        error_primal_displacement.append(np.linalg.norm(
            primal_solutions["value"][i][:n_dofs["space"]] - projected_reduced_solutions["value"][i][:n_dofs["space"]]) / np.linalg.norm(primal_solutions["value"][i][:n_dofs["space"]]))
        error_primal_velo.append(np.linalg.norm(
            primal_solutions["value"][i][n_dofs["space"]:] - projected_reduced_solutions["value"][i][n_dofs["space"]:]) / np.linalg.norm(primal_solutions["value"][i][n_dofs["space"]:]))
    else:
        error_primal_displacement.append(0)
        error_primal_velo.append(0)
    if primal_solutions["value"][i][index2measuredisp] != 0.0:
        error_primal_displacement_pointwise.append((primal_solutions["value"][i][index2measuredisp]-projected_reduced_solutions["value"][i][index2measuredisp])/primal_solutions["value"][i][index2measuredisp])
    else:
        error_primal_displacement_pointwise.append(0.0)


# %% Plotting
time_step_size = 40.0 / (slab_properties["n_total"])


# plot pointwise displacement at (6,0.5,0)
# plt.rc('text', usetex=True)
# plt.rcParams["figure.figsize"] = (10,2)
plt.plot(np.arange(0, slab_properties["n_total"]*n_dofs["solperstep"]+1),
         tuple(primal_solutions["value"][i][index2measuredisp] for i in range(slab_properties["n_total"]*n_dofs["solperstep"]+1)), label="pw displacement -fom")
plt.plot(np.arange(0, slab_properties["n_total"]*n_dofs["solperstep"]+1),
         tuple(projected_reduced_solutions["value"][i][index2measuredisp] for i in range(n_dofs["solperstep"]*slab_properties["n_total"]+1)), label="pw displacement -rom")
# plt.plot(np.arange(0, slab_properties["n_total"]+1),
#          error_dual_displacement, label="dual")
# plt.yscale('log')
plt.legend()
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.show()





# plot temporal evolution of cost functionals  s
# plt.rc('text', usetex=True)
# plt.rcParams["figure.figsize"] = (10,2)
plt.plot(np.vstack(projected_reduced_solutions_slab["time"])[:,-1],
          J_h_t, color='r', label="$u_h$")
plt.plot(np.vstack(projected_reduced_solutions_slab["time"])[:,-1],
          J_r_t, '--', c='#1f77b4', label="$u_N$")
plt.grid()
plt.legend()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.xlabel('$t [$s$]$')
plt.ylabel("$J(u)$")
plt.xlim([0, slab_properties["n_total"]*time_step_size])
plt.show()


# plot temporal evolution of error and error estimate
plt.rcParams["figure.figsize"] = (10,2)
plt.plot(np.vstack(projected_reduced_solutions_slab["time"])[:,-1],
          np.array(temporal_interval_error), c='#1f77b4', label="estimate")
# plt.plot(np.vstack(projected_reduced_solutions_slab["time"])[:,-1],
#           np.array(temporal_interval_error_shorti), c='black', label="estimate")
plt.plot(np.vstack(projected_reduced_solutions_slab["time"])[:,-1],
          np.abs(J_h_t-J_r_t), color='r', label="error")
plt.grid()
plt.legend()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.xlabel('$t [$s$]$')
plt.ylabel("$error$")
plt.xlim([0, slab_properties["n_total"]*time_step_size])

plt.show()



# %%
