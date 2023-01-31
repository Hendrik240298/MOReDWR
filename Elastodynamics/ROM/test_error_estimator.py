import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.linalg
import scipy.interpolate
from scipy.sparse import coo_matrix, bmat, lil_matrix
import matplotlib.pyplot as plt
import os
import time
import sys
from iPOD import iPOD, ROM_update, ROM_update_dual, reduce_matrix, reduce_vector, project_vector
from auxiliaries import save_vtk, read_in_LES, apply_boundary_conditions, read_in_discretization, solve_primal_FOM_step, solve_dual_FOM_step, solve_primal_ROM_step, reorder_matrix, reorder_vector, error_estimator, save_solution_txt, load_solution_txt, evaluate_cost_functional, find_solution_indicies_for_slab
#import imageio

PLOTTING = False
MOTHER_PATH = "/home/hendrik/Code/MORe_DWR/Elastodynamics/"
OUTPUT_PATH = MOTHER_PATH + "/Data/3D/Rod/"
OUTPUT_PATH_DUAL = MOTHER_PATH + "Dual_Elastodynamics/Data/3D/Rod/"
cycle = "cycle=1"
SAVE_PATH = MOTHER_PATH + "Data/ROM/" + cycle + "/"

LOAD_SOLUTION = False

# if SAVE_PATH directory not exists create it
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)


print(f"\n{'-'*12}\n| {cycle}: |\n{'-'*12}\n")

ENERGY_DUAL = 0.999999
ENERGY_PRIMAL = {"displacement": 0.99999999,
                 "velocity":     0.99999999}

# %% read in properties connected to discretization
n_dofs, slab_properties, index2measuredisp, dof_matrix, grid = read_in_discretization(
    OUTPUT_PATH + cycle)

# # modify system shortining
# slab_properties["n_total"] = int(slab_properties["n_total"]*2/3)
# slab_properties["time_points"] =  slab_properties["time_points"][:slab_properties["n_total"]]
# print(slab_properties['n_total'])

# %% Reading in matricies and rhs without bc
matrix_no_bc, rhs_no_bc = read_in_LES(
    OUTPUT_PATH + cycle, "/matrix_no_bc.txt", "primal_rhs_no_bc")
mass_matrix_no_bc, _ = read_in_LES(
    OUTPUT_PATH + cycle, "/mass_matrix_no_bc.txt", "primal_rhs_no_bc")

_, dual_rhs_no_bc = read_in_LES(
    OUTPUT_PATH + cycle, "/matrix_no_bc.txt", "dual_rhs_no_bc")

# dual matrix is primal.T + mass_matrix.T
dual_matrix_no_bc = matrix_no_bc.T + mass_matrix_no_bc.T


# # modify system shortining
# rhs_no_bc = rhs_no_bc[:slab_properties["n_total"]]
# dual_rhs_no_bc = dual_rhs_no_bc[:slab_properties["n_total"]]

# %% Enforcing BC to primal und dual systems
primal_matrix, primal_system_rhs = apply_boundary_conditions(
    matrix_no_bc, rhs_no_bc, OUTPUT_PATH + cycle + "/boundary_id.txt")
dual_matrix, dual_system_rhs = apply_boundary_conditions(
    dual_matrix_no_bc, dual_rhs_no_bc, OUTPUT_PATH + cycle + "/boundary_id.txt")

# %% read in IC
initial_solution = np.loadtxt(OUTPUT_PATH + cycle + "/initial_solution.txt")
# modify system shortining
# print(SAVE_PATH + f"/py_solution0{1000:4}.txt")
# initial_solution = np.loadtxt(SAVE_PATH + "/initial_cond_mid.txt")
# # set velocity to zero
# initial_solution[n_dofs["space"]:2 * n_dofs["space"]] = 0.0
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

last_time_step_mass_matrix_no_bc = lil_matrix(
    (mass_matrix_no_bc.shape[0], mass_matrix_no_bc.shape[1]))
print(
    last_time_step_mass_matrix_no_bc[-n_dofs["time_step"]:, -n_dofs["time_step"]:].shape)
print(mass_matrix_no_bc[:n_dofs["time_step"], :n_dofs["time_step"]].shape)

last_time_step_mass_matrix_no_bc[:n_dofs["time_step"], -n_dofs["time_step"]:] = mass_matrix_no_bc[:n_dofs["time_step"], :n_dofs["time_step"]]

print(f"A.shape = {A.shape}")
print(f"B.shape = {B.shape}")
print(f"C.shape = {C.shape}")
print(f"D.shape = {D.shape}")
print(f"matrix_no_bc.shape = {matrix_no_bc.shape}")

# ------------
# %% Primal FOM solve
start_execution = time.time()


primal_solutions = {"value": [initial_solution], "time": [0.]}

for i in range(slab_properties["n_total"]):
    primal_solutions = solve_primal_FOM_step(
        primal_solutions, D_wbc, C_wbc, primal_system_rhs[i], slab_properties, n_dofs, i)

end_execution = time.time()

execution_time_FOM = end_execution - start_execution

print("Primal FOM time:   " + str(execution_time_FOM))
print("n_dofs[space] =", n_dofs["space"])

for i, primal_solution in enumerate(primal_solutions["value"]):
        save_vtk(SAVE_PATH + f"/py_solution{i:05}.vtk", {"displacement": dof_matrix.dot(primal_solution[0:n_dofs["space"]]), "velocity": dof_matrix.dot(
            primal_solution[n_dofs["space"]:2 * n_dofs["space"]])}, grid, cycle=i, time=primal_solutions["time"][i])

# %% Primal slab editor
primal_solutions_slab = {"value": [], "time": []}
for i in range(slab_properties["n_total"]):
    primal_solutions_slab["time"].append(slab_properties["time_points"][i])
    indices = find_solution_indicies_for_slab(primal_solutions_slab["time"][-1], primal_solutions["time"])
    primal_solutions_slab["value"].append(np.hstack([primal_solutions["value"][i] for i in indices]).T)


# %% dual FOM solve
start_execution = time.time()
last_dual_solution = np.zeros((n_dofs["time_step"],))

# for debugging:
# dual_system_rhs = primal_system_rhs[::-1]
dual_solutions = {"value": [last_dual_solution],
                  "time": [slab_properties["time_points"][-1][-1]]}

for i in list(range(slab_properties["n_total"]))[::-1]:
    dual_solutions = solve_dual_FOM_step(
        dual_solutions, A_wbc, B_wbc, dual_system_rhs[i], slab_properties, n_dofs, i)


dual_solutions["value"] = dual_solutions["value"][::-1]
dual_solutions["time"] = dual_solutions["time"][::-1]

end_execution = time.time()
execution_time_FOM = end_execution - start_execution
print("Dual FOM time:   " + str(execution_time_FOM))


dual_solutions_slab = {"value": [], "time": []}
for i in range(slab_properties["n_total"]):
    dual_solutions_slab["time"].append(slab_properties["time_points"][i])
    indices = find_solution_indicies_for_slab(dual_solutions_slab["time"][-1], dual_solutions["time"])
    dual_solutions_slab["value"].append(np.hstack([dual_solutions["value"][i] for i in indices]).T)



# %% Definition and initialization of ROM ingredients

bunch_size = 1  # len(primal_solutions)  # 1

total_energy = {"displacement": 0, "velocity": 0}
pod_basis = {"displacement": np.empty([0, 0]), "velocity": np.empty([0, 0])}
bunch = {"displacement": np.empty([0, 0]), "velocity": np.empty([0, 0])}
singular_values = {"displacement": np.empty(
    [0, 0]), "velocity": np.empty([0, 0])}


for primal_solution in primal_solutions["value"][0:400]:
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


print(pod_basis["displacement"].shape[1])
print(pod_basis["velocity"].shape[1])

# compute reduced matrices
C_reduced = reduce_matrix(C, pod_basis, pod_basis)
D_reduced = reduce_matrix(D, pod_basis, pod_basis)

# %% Primal ROM solve
# reduced_solutions = {"value": [], "time": []}

reduced_solutions = []
reduced_solution_old = reduce_vector(initial_solution[:], pod_basis)

projected_reduced_solutions = {"value": [project_vector(reduce_vector(initial_solution[:], pod_basis), pod_basis)],
                               "time": [0.]}


perfect_reduced_solutions = {"value": [primal_solutions["value"][0]],
                             "time": [0.]}


last_projected_reduced_solution = projected_reduced_solutions
projected_reduced_solutions_before_enrichment = []


temporal_interval_error = []
temporal_interval_error_shorti = []
temporal_interval_error_incidactor = []
temporal_interval_error_relative = []

temporal_residual_norm = []
temporal_dual_norm = [] 
temporal_solution_diff_norm = []

goal_func_error = []

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

n_dofs["reduced_primal"] = pod_basis["displacement"].shape[1] + \
    pod_basis["velocity"].shape[1]

start_time = time.time()
for i in range(slab_properties["n_total"]):

    projected_reduced_solutions, reduced_solution_old = solve_primal_ROM_step(projected_reduced_solutions, reduced_solution_old, D_reduced, C_reduced, rhs_no_bc[i], pod_basis, slab_properties, n_dofs, i)

    # override rom with fom solution
    # projected_reduced_solutions["value"] = primal_solutions["value"][:len(projected_reduced_solutions["value"])]
    # projected_reduced_solutions["time"] = primal_solutions["time"][:len(projected_reduced_solutions["time"])]
    # reduced_solution_old = reduce_vector(projected_reduced_solutions["value"][-1], pod_basis)
    
    # temporal_interval_error.append(error_estimator(projected_reduced_solutions, dual_solutions, matrix_no_bc, rhs_no_bc[i].copy(), slab_properties))

    # print(i)
    projected_slab = {"value": [], "time": []}
    dual_projected_slab = {"value": [], "time": []}

    index = find_solution_indicies_for_slab(slab_properties["time_points"][i], projected_reduced_solutions["time"])

    projected_slab["value"] = np.hstack([projected_reduced_solutions["value"][k] for k in index])
    projected_slab["time"] = np.hstack([projected_reduced_solutions["time"][k] for k in index])

    # projected_slab["value"] = np.hstack(projected_reduced_solutions["value"][-slab_properties["n_time_unknowns"]-1:])
    # projected_slab["time"] = np.hstack(projected_reduced_solutions["time"][-slab_properties["n_time_unknowns"]-1:])


    # find argument where time of dual_projected_reduced_solutions is equal to time of projected_slab
    index_of_dual = find_solution_indicies_for_slab(projected_slab["time"], dual_solutions["time"])
   
    # for j in range(len(projected_slab["time"])):
    #     for k in range(len(dual_solutions["time"])):
    #         if dual_solutions["time"][k] == projected_slab["time"][j]:
    #             index_of_dual.append(k)
        

    # hstack last entries of projected_reduced_solutions to obtain slab
    dual_projected_slab["value"] = np.hstack(
        [dual_solutions["value"][k] for k in index_of_dual])

    # hstack i-th last entries of projected_reduced_solutions["time"] to obtain slab
    dual_projected_slab["time"] = np.hstack(
        [dual_solutions["time"][k] for k in index_of_dual])

    # dual_projected_slab["value"] = dual_solutions_slab["value"][i].copy()
    # projected_slab["value"] = primal_solutions_slab["value"][i].copy()
    
    residual_slab = - matrix_no_bc.dot(projected_slab["value"]) + rhs_no_bc[i].copy()
    # residual_slab -= mass_matrix_no_bc.dot(projected_slab["value"])

    # residual_slab = - matrix_no_bc.dot(primal_solutions_slab["value"][i]) + rhs_no_bc[i].copy()
    # residual_slab = matrix_no_bc.dot(primal_solutions_slab["value"][i]-projected_slab["value"])
    
    # print(f"res: {np.linalg.norm(residual_slab)},  dual: {np.linalg.norm(dual_projected_slab['value'])}")
    temporal_residual_norm.append(np.linalg.norm(residual_slab))
    temporal_dual_norm.append(np.linalg.norm(dual_projected_slab["value"]))
    
    temporal_solution_diff_norm.append(np.linalg.norm(primal_solutions_slab["value"][i] - projected_slab["value"]))
    goal_func_error.append((primal_solutions_slab["value"][i] - projected_slab["value"]).dot(dual_rhs_no_bc[i]))

    temporal_interval_error.append(
        (np.dot(dual_projected_slab["value"], residual_slab)))


    if i<slab_properties["n_total"]-2:
        index_of_next_dual = find_solution_indicies_for_slab(slab_properties["time_points"][i+1], dual_solutions["time"])

        next_projected_dual_slab = {"value": [], "time": []}
        next_projected_dual_slab["value"] = np.hstack(
            [dual_solutions["value"][k] for k in index_of_next_dual])
        next_projected_dual_slab["time"] = np.hstack(
            [dual_solutions["time"][k] for k in index_of_next_dual])

    delta_sol = primal_solutions_slab["value"][i] - projected_slab["value"]

    lhs = (delta_sol).dot(dual_rhs_no_bc[i])
    rhs =  matrix_no_bc.dot(delta_sol)
    rhs += mass_matrix_no_bc.dot(delta_sol)
    old_sol_rhs = 0.
    if i<slab_properties["n_total"]-2:
        old_sol_rhs = mass_matrix_no_bc.dot(delta_sol).dot(next_projected_dual_slab["value"])
    rhs  = rhs.dot(dual_projected_slab["value"]) - old_sol_rhs

    print(f"lhs: {lhs}, rhs: {rhs}")



    last_projected_slab = {"value": [], "time": []}
    last_projected_slab["value"] = projected_slab["value"].copy()
    
    last_projected_dual_slab = {"value": [], "time": []}
    last_projected_dual_slab["value"] = dual_projected_slab["value"].copy()
    

    # if i > 0:
    #     print(np.linalg.norm(projected_slab["value"][:n_dofs["time_step"]
    #                                              ] - last_projected_slab["value"][-n_dofs["time_step"]:]))

    # # add initial condition to residual
    # residual_slab -= mass_matrix_no_bc.dot(projected_slab["value"])
    # # if slab is not first slab: residual_slab += last_time_step_mass_matrix_no_bx.dot(last_projected_slab["value"]))
    # if i > 0:
    #     residual_slab += last_time_step_mass_matrix_no_bc.dot(
    #         last_projected_slab["value"])
    
end_execution = time.time()
execution_time_ROM = end_execution - start_execution
print("ROM time:        " + str(execution_time_ROM))

# generate slabs for projected reduced solutions
projected_reduced_solutions_slab = {"value": [], "time": []}
for i in range(slab_properties["n_total"]):
    projected_reduced_solutions_slab["value"].append(np.hstack(
        projected_reduced_solutions["value"][i*(n_dofs['solperstep']): (i+1)*(n_dofs['solperstep'])+1]).T)
    projected_reduced_solutions_slab["time"].append(
        slab_properties["time_points"][i])

# %% Computing reduced and full Cost functional

J_r_t = np.empty([slab_properties["n_total"], 1])
J_h_t = np.empty([slab_properties["n_total"], 1])
# TODO: is this correct? Yes, julian it is correct ;)
for i in range(slab_properties["n_total"]):
    J_r_t[i] = projected_reduced_solutions_slab["value"][i].dot(dual_rhs_no_bc[i])
    J_h_t[i] = primal_solutions_slab["value"][i].dot(dual_rhs_no_bc[i])

true_error = np.abs(np.sum(J_h_t-J_r_t))
true_abs_error = np.sum(np.abs(J_h_t-J_r_t))
estimated_error = np.abs(np.sum(temporal_interval_error))
estimated_abs_error = np.sum(np.abs(temporal_interval_error))
efficiency = true_error/estimated_error
print("true error:          " + str(true_error))
print("estimated error:     " + str(estimated_error))
print("efficiency:          " + str(efficiency))
print(" ")
print("true abs error:      " + str(true_abs_error))
print("estimated abs error: " + str(estimated_abs_error))
print("efficiency abs:      " + str(true_abs_error/estimated_abs_error))

# %% Plotting
time_step_size = 40.0 / (slab_properties["n_total"])


plt.plot(np.vstack(projected_reduced_solutions_slab["time"])[:, -1],
        temporal_solution_diff_norm, color='r', label="temporal_solution_diff_norm")
plt.grid()
plt.legend()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.xlabel('$t [$s$]$')
plt.ylabel("norm")
plt.xlim([0, slab_properties["n_total"]*time_step_size])
plt.show()


# plt.plot(np.vstack(projected_reduced_solutions_slab["time"])[:, -1],
#         [np.linalg.norm(dual_rhs_no_bc[i]) for i in range(len(dual_rhs_no_bc))], color='r', label="dual_rhs_no_bc")
# plt.grid()
# plt.legend()
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
# plt.xlabel('$t [$s$]$')
# plt.ylabel("norm")
# plt.xlim([0, slab_properties["n_total"]*time_step_size])
# plt.show()


plt.plot(np.vstack(projected_reduced_solutions_slab["time"])[:, -1],
        temporal_residual_norm, color='r', label="residual")
plt.plot(np.vstack(projected_reduced_solutions_slab["time"])[:, -1],
          temporal_dual_norm, '--', c='#1f77b4', label="dual")
plt.grid()
plt.legend()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.xlabel('$t [$s$]$')
plt.ylabel("norm")
plt.xlim([0, slab_properties["n_total"]*time_step_size])
plt.show()

plt.plot(np.vstack(projected_reduced_solutions_slab["time"])[:, -1],
        goal_func_error, color='r', label="goal_func_error")
plt.grid()
plt.legend()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.xlabel('$t [$s$]$')
plt.ylabel("norm")
plt.xlim([0, slab_properties["n_total"]*time_step_size])
plt.show()

# pointwise displacement
plt.plot(np.arange(0, slab_properties["n_total"]*n_dofs["solperstep"]+1),
         tuple(primal_solutions["value"][i][index2measuredisp] for i in range(slab_properties["n_total"]*n_dofs["solperstep"]+1)), label="pw displacement -fom")
plt.plot(np.arange(0, slab_properties["n_total"]*n_dofs["solperstep"]+1),
         tuple(projected_reduced_solutions["value"][i][index2measuredisp] for i in range(n_dofs["solperstep"]*slab_properties["n_total"]+1)), label="pw displacement -rom")
plt.legend()
plt.show()


# Cost functional
plt.plot(np.vstack(projected_reduced_solutions_slab["time"])[:, -1],
         J_h_t, color='r', label="$u_h$")
plt.plot(np.vstack(projected_reduced_solutions_slab["time"])[:, -1],
         J_r_t, '--', c='#1f77b4', label="$u_N$")
plt.grid()
plt.legend()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.xlabel('$t [$s$]$')
plt.ylabel("$J(u)$")
plt.xlim([0, slab_properties["n_total"]*time_step_size])
plt.show()


# plot temporal evolution of error and error estimate
plt.rcParams["figure.figsize"] = (10, 6)
plt.plot(np.vstack(projected_reduced_solutions_slab["time"])[:, -1],
         np.array(temporal_interval_error), c='#1f77b4', label="estimate")
plt.plot(np.vstack(projected_reduced_solutions_slab["time"])[:, -1],
         (J_h_t-J_r_t), color='r', label="error")
plt.grid()
plt.legend()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.xlabel('$t [$s$]$')
plt.ylabel("$error$")
plt.xlim([0, slab_properties["n_total"]*time_step_size])

plt.show()
