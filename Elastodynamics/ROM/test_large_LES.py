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
from auxiliaries import save_vtk, read_in_LES, apply_boundary_conditions, read_in_discretization
from auxiliaries import solve_primal_FOM_step, solve_dual_FOM_step, solve_primal_ROM_step, reorder_matrix
from auxiliaries import reorder_vector, error_estimator, save_solution_txt, load_solution_txt 
from auxiliaries import evaluate_cost_functional, find_solution_indicies_for_slab, plot_matrix
#import imageio

"""
We try to test if the error estimator works when consistency using large LES
* Highlights test 
! Right now even the primal FOM does not work
? What is the problem?
TODO Try to repair primal FOM
"""


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
ENERGY_PRIMAL = {"displacement": 0.999999,
                 "velocity":     0.999999}

# %% read in properties connected to discretization
n_dofs, slab_properties, index2measuredisp, dof_matrix, grid = read_in_discretization(
    OUTPUT_PATH + cycle)


# %% Reading in matrices and rhs without bc
matrix_no_bc, rhs_no_bc = read_in_LES(
    OUTPUT_PATH + cycle, "/matrix_no_bc.txt", "primal_rhs_no_bc")
mass_matrix_no_bc, _ = read_in_LES(
    OUTPUT_PATH + cycle, "/mass_matrix_no_bc.txt", "primal_rhs_no_bc")

_, dual_rhs_no_bc = read_in_LES(
    OUTPUT_PATH + cycle, "/matrix_no_bc.txt", "dual_rhs_no_bc")

matrix_no_bc_for_dual = matrix_no_bc.copy()
"""
* A := mass_matrix_no_bc

* matrix_no_bc * U = rhs_no_bc
* <=> 
* (-A/2 + 2*B*k/15,    2*A/3 + B*k/15,    -A/6 - B*k/30) (U_n)      (F, phi_n+1/2) 
* (-2*A/3 + B*k/15,     8*B*k/15,       2*A/3 + B*k/15) (U_n+1/2) = (F, phi_n+1/2)
* (A/6 - B*k/30,     -2*A/3 + B*k/15,   A/2 + 2*B*k/15) (U_n+1)     (F, phi_n+1)

! -------------------------------------------------------------------------------

* [matrix_no_bc + mass_matrix_no_bc] U = rhs 
* <=> 
* (A/2 + 2*B*k/15,    2*A/3 + B*k/15,    -A/6 - B*k/30) (U_n)       (F, phi_n+1/2) + U_old A
* (-2*A/3 + B*k/15,     8*B*k/15,       2*A/3 + B*k/15) (U_n+1/2) = (F, phi_n+1/2)
* (A/6 - B*k/30,     -2*A/3 + B*k/15,   A/2 + 2*B*k/15) (U_n+1)     (F, phi_n+1)
"""

# * System Matrix = system_matrix + weight_mass_matrix * mass_matrix
matrix_no_bc = matrix_no_bc + mass_matrix_no_bc

mass_matrix_up_right_no_bc = np.zeros((mass_matrix_no_bc.shape[0],mass_matrix_no_bc.shape[1]))
mass_matrix_up_right_no_bc[:n_dofs["time_step"], -n_dofs["time_step"]:] = mass_matrix_no_bc[:n_dofs["time_step"], :n_dofs["time_step"]].toarray()
mass_matrix_up_right_no_bc = scipy.sparse.csr_matrix(mass_matrix_up_right_no_bc)

# * lower diagonal
mass_matrix_down_right_no_bc = np.zeros((mass_matrix_no_bc.shape[0],mass_matrix_no_bc.shape[1]))
mass_matrix_down_right_no_bc[-n_dofs["time_step"]:, -n_dofs["time_step"]:] = mass_matrix_no_bc[:n_dofs["time_step"], :n_dofs["time_step"]].toarray()
mass_matrix_down_right_no_bc = scipy.sparse.csr_matrix(mass_matrix_down_right_no_bc)

# * left down corner --> transposed for dual LES
mass_matrix_down_left_no_bc = np.zeros((mass_matrix_no_bc.shape[0],mass_matrix_no_bc.shape[1]))
mass_matrix_down_left_no_bc[-n_dofs["time_step"]:, :n_dofs["time_step"]] = mass_matrix_no_bc[:n_dofs["time_step"], :n_dofs["time_step"]].toarray().T
mass_matrix_down_left_no_bc = scipy.sparse.csr_matrix(mass_matrix_down_left_no_bc)

# ? Apply mass_matrix to last time step? Thus diag(mass_matrix) = [0, 0, M.T] instead of [M.T, 0, 0]
dual_matrix_no_bc = matrix_no_bc_for_dual.T + mass_matrix_no_bc.T

# print(type(matrix_no_bc))
# print(type(matrix_no_bc_for_dual))
# dual_matrix_no_bc = scipy.sparse.csr_matrix(matrix_no_bc_for_dual.T + mass_matrix_down_right_no_bc.T)


# %% Enforcing BC to primal und dual systems
primal_matrix, primal_system_rhs = apply_boundary_conditions(
    matrix_no_bc, rhs_no_bc, OUTPUT_PATH + cycle + "/boundary_id.txt")

print(type(dual_matrix_no_bc))

dual_matrix, dual_system_rhs = apply_boundary_conditions(
    dual_matrix_no_bc, dual_rhs_no_bc, OUTPUT_PATH + cycle + "/boundary_id.txt")

# ? do i really dont need this? --> In Paraview it seems like the application of BC removes the oscillation in solution
mass_matrix_up_right, _ = apply_boundary_conditions(
    mass_matrix_up_right_no_bc, rhs_no_bc, OUTPUT_PATH + cycle + "/boundary_id.txt")

mass_matrix_up_down_right, _ = apply_boundary_conditions(
    mass_matrix_down_right_no_bc, rhs_no_bc, OUTPUT_PATH + cycle + "/boundary_id.txt")

mass_matrix_down_left, _ = apply_boundary_conditions(
    mass_matrix_down_left_no_bc, rhs_no_bc, OUTPUT_PATH + cycle + "/boundary_id.txt")
# %% read in IC
initial_solution = np.loadtxt(OUTPUT_PATH + cycle + "/initial_solution.txt")

# %% Primal FOM solve ------------------------------------------------------------------------
# ! Benchmarked in Paraview against deal.ii solution --> minor errors thus assuming that solver is correct

SKIP_PRIMAL = False


slab_properties["n_total"] = 100

if not SKIP_PRIMAL:
    start_execution = time.time()

    primal_solutions_slab = {"value": [], "time": []}
    for i in range(slab_properties["n_total"]):
        if i == 0:
            primal_rhs = primal_system_rhs[i].copy() + mass_matrix_up_right.dot(np.zeros(primal_matrix.shape[0]))
        else:
            primal_rhs = primal_system_rhs[i].copy() + mass_matrix_up_right.dot(primal_solutions_slab["value"][-1])
        primal_solution = scipy.sparse.linalg.spsolve(primal_matrix, primal_rhs)
        
        primal_solutions_slab["value"].append(primal_solution)
        primal_solutions_slab["time"].append(slab_properties["time_points"][i])

    end_execution = time.time()

    execution_time_FOM = end_execution - start_execution

    print("Primal FOM time:   " + str(execution_time_FOM))
    print("n_dofs[space]:     ", n_dofs["space"])
    print("time steps:        ", slab_properties["n_total"])

    primal_solutions = {"value": [np.zeros(int(n_dofs["time_step"]))], "time": [0.]}

    for i, primal_solution_slab in enumerate(primal_solutions_slab["value"]):
        for j in range(len(slab_properties["time_points"][i][1:])):
            range_start = int((j)*n_dofs["time_step"])
            range_end = int((j+1)*n_dofs["time_step"])
            primal_solutions["value"].append(primal_solutions_slab["value"][i][range_start:range_end])
            primal_solutions["time"].append(slab_properties["time_points"][i][j+1])

    for i, primal_solution in enumerate(primal_solutions["value"]):
            save_vtk(SAVE_PATH + f"/py_solution{i:05}.vtk", {"displacement": dof_matrix.dot(primal_solution[0:n_dofs["space"]]), "velocity": dof_matrix.dot(
                primal_solution[n_dofs["space"]:2 * n_dofs["space"]])}, grid, cycle=i, time=primal_solutions["time"][i])



# %% dual FOM solve ------------------------------------------------------------------------
start_execution = time.time()
last_dual_solution = np.zeros((n_dofs["time_step"],))

# for debugging:
# dual_system_rhs = primal_system_rhs[::-1]
dual_solutions_slab = {"value": [],
                  "time": []}

for i in list(range(slab_properties["n_total"]))[::-1]:
    if i == len(list(range(slab_properties["n_total"])))-1:
        print("I was here")
        dual_rhs = dual_system_rhs[i].copy() + mass_matrix_down_left.dot(np.zeros(dual_matrix.shape[0]))
    else:
        dual_rhs = dual_system_rhs[i].copy() + mass_matrix_down_left.dot(dual_solutions_slab["value"][-1])
    dual_solution =  scipy.sparse.linalg.spsolve(dual_matrix, dual_rhs)
    
    # if i != len(list(range(slab_properties["n_total"])))-1:
    #     print(i, np.linalg.norm(dual_solution[-n_dofs["time_step"]:] - dual_solutions_slab["value"][-1][:n_dofs["time_step"]]))
    dual_solutions_slab["value"].append(dual_solution)
    dual_solutions_slab["time"].append(slab_properties["time_points"][i])
    # print index i and time
    # print(i, slab_properties["time_points"][i])


dual_solutions_slab["value"] = dual_solutions_slab["value"][::-1]
dual_solutions_slab["time"] = dual_solutions_slab["time"][::-1]

end_execution = time.time()
execution_time_FOM = end_execution - start_execution
print("Dual FOM time:   " + str(execution_time_FOM))


dual_solutions = {"value": [np.zeros(int(n_dofs["time_step"]))], "time": [0.]}

for i, dual_solution_slab in enumerate(dual_solutions_slab["value"]):
    for j in range(len(slab_properties["time_points"][i][1:])):
        range_start = int((j)*n_dofs["time_step"])
        range_end = int((j+1)*n_dofs["time_step"])
        dual_solutions["value"].append(dual_solutions_slab["value"][i][range_start:range_end])
        dual_solutions["time"].append(slab_properties["time_points"][i][j+1])

for i, dual_solution in enumerate(dual_solutions["value"]):
        save_vtk(SAVE_PATH + f"/py_dual_solution{i:05}.vtk", {"displacement": dof_matrix.dot(dual_solution[0:n_dofs["space"]]), "velocity": dof_matrix.dot(
            dual_solution[n_dofs["space"]:2 * n_dofs["space"]])}, grid, cycle=i, time=dual_solutions["time"][i])



# %% Definition and initialization of ROM ingredients

bunch_size = 1  # len(primal_solutions)  # 1

total_energy = {"displacement": 0, "velocity": 0}
pod_basis = {"displacement": np.empty([0, 0]), "velocity": np.empty([0, 0])}
bunch = {"displacement": np.empty([0, 0]), "velocity": np.empty([0, 0])}
singular_values = {"displacement": np.empty(
    [0, 0]), "velocity": np.empty([0, 0])}


for primal_solution in primal_solutions["value"]:# [0:400]:
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


# * reduced matrices
mass_matrix_up_right_reduced = reduce_matrix(mass_matrix_up_right_no_bc, pod_basis, pod_basis)
system_matrix_reduced = reduce_matrix(matrix_no_bc, pod_basis, pod_basis)
# ------------------------------------------------------------------------------------------
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

# alpha = 1.1
projected_reduced_solutions_slab = {"value": [], "time": []}

for i in range(slab_properties["n_total"]):

    # solve really the rom system but full 3x3 block system
    if i == 0:
        primal_rhs_reduced = reduce_vector(primal_system_rhs[i].copy(), pod_basis) + mass_matrix_up_right_reduced.dot(np.zeros(system_matrix_reduced.shape[0]))
    else:
        primal_rhs_reduced = reduce_vector(primal_system_rhs[i].copy(), pod_basis) + mass_matrix_up_right_reduced.dot(primal_solution_reduced)
    # primal_solution_reduced = scipy.sparse.linalg.spsolve(system_matrix_reduced, primal_rhs_reduced)
    primal_solution_reduced = np.linalg.solve(system_matrix_reduced, primal_rhs_reduced)
        
    projected_reduced_solutions_slab["value"].append(project_vector(primal_solution_reduced, pod_basis))
    projected_reduced_solutions_slab["time"].append(slab_properties["time_points"][i])

    dual_projected_slab = {"value": [], "time": []}
    
    dual_projected_slab["value"] = dual_solutions_slab["value"][i] #[i]
    # projected_slab["value"] = alpha*primal_solutions_slab["value"][i]
    
    if i > 0:
        # residual_slab = - matrix_no_bc.dot(projected_reduced_solutions_slab["value"][-1]) + rhs_no_bc[i].copy() + mass_matrix_up_right_no_bc.dot(projected_reduced_solutions_slab["value"][-2])
        residual_slab = - matrix_no_bc_for_dual.dot(projected_reduced_solutions_slab["value"][-1]) + rhs_no_bc[i].copy()
    else:
        residual_slab = - matrix_no_bc.dot(projected_reduced_solutions_slab["value"][-1]) + rhs_no_bc[i].copy() 
    
    temporal_residual_norm.append(np.linalg.norm(residual_slab))
    temporal_dual_norm.append(np.linalg.norm(dual_projected_slab["value"]))
    if np.linalg.norm(residual_slab) < 1e-6:
        print(f"residual norm is zero at {i}")
    if np.linalg.norm(dual_projected_slab["value"]) < 1e-6:
        print(f"dual norm is zero at {i}")
        
    # temporal_interval_error.append((np.dot(dual_projected_slab["value"], residual_slab)))  # TODO write as residual_slab.dot(dual_projected_slab["value"])
    temporal_interval_error.append( dual_projected_slab["value"].dot(residual_slab))  # TODO write as residual_slab.dot(dual_projected_slab["value"])

    temporal_solution_diff_norm.append(np.linalg.norm(primal_solutions_slab["value"][i] - projected_reduced_solutions_slab["value"][-1])/np.linalg.norm(primal_solutions_slab["value"][i]))
    goal_func_error.append((primal_solutions_slab["value"][i] - projected_reduced_solutions_slab["value"][-1]).dot(dual_rhs_no_bc[i]))



end_execution = time.time()
execution_time_ROM = end_execution - start_execution
print("ROM time:        " + str(execution_time_ROM))

# generate slabs for projected reduced solutions
# projected_reduced_solutions_slab = {"value": [], "time": []}

# for i in range(slab_properties["n_total"]):
#     # projected_reduced_solutions_slab["value"].append(np.hstack(projected_reduced_solutions["value"][i*(n_dofs['solperstep']): (i+1)*(n_dofs['solperstep'])+1]).T)
#     projected_reduced_solutions_slab["value"].append(alpha*primal_solutions_slab["value"][i])
#     projected_reduced_solutions_slab["time"].append(slab_properties["time_points"][i])
#     # print(f"{i*(n_dofs['solperstep'])} - {(i+1)*(n_dofs['solperstep'])+1} for times {slab_properties['time_points'][i]}")

# %% Computing reduced and full Cost functional

J_r_t = np.empty([slab_properties["n_total"], 1])
J_h_t = np.empty([slab_properties["n_total"], 1])
# TODO: is this correct? Yes, julian it is correct ;)
for i in range(slab_properties["n_total"]):
    J_r_t[i] = projected_reduced_solutions_slab["value"][i].dot(dual_rhs_no_bc[i])
    J_h_t[i] = primal_solutions_slab["value"][i].dot(dual_rhs_no_bc[i])

print("J_r(0): " + str(J_r_t[0]))
print("J_h(0): " + str(J_h_t[0]))
true_error = np.abs(np.sum(J_h_t-J_r_t))
true_abs_error = np.sum(np.abs(J_h_t-J_r_t))
estimated_error = np.abs(np.sum(temporal_interval_error))
estimated_abs_error = np.sum(np.abs(temporal_interval_error))
efficiency = true_error/estimated_error
print("J_h:                 " + str(np.sum(J_h_t)))
print("J_r:                 " + str(np.sum(J_r_t)))
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
# plt.xlim([0, slab_properties["n_total"]*time_step_size])
plt.savefig('images/temporal_solution_diff_norm.png')

plt.show()



# create figure and axis objects with subplots()
fig,ax = plt.subplots()
# make a plot
ax.plot(np.vstack(projected_reduced_solutions_slab["time"])[:, -1],
        temporal_residual_norm, color='r', label="temporal_residual_norm")
ax.set_ylabel("residual norm")
ax.grid()
ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(np.vstack(projected_reduced_solutions_slab["time"])[:, -1], temporal_dual_norm, c='#1f77b4', label="temporal_dual_norm")
ax2.set_ylabel("dual norm")
plt.show()
fig.savefig('images/norm_dual_and_res.png')


# plt.plot(np.vstack(projected_reduced_solutions_slab["time"])[:, -1],
        
# plt.plot(np.vstack(projected_reduced_solutions_slab["time"])[:, -1],
#         temporal_dual_norm, c='#1f77b4', label="temporal_dual_norm")
# plt.grid()
# plt.legend()
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
# plt.xlabel('$t [$s$]$')
# plt.ylabel("norm")
# plt.xlim([0, slab_properties["n_total"]*time_step_size])
# plt.savefig('images/norm_dual_and_res.png')

# plt.show()


# plt.plot(np.vstack(projected_reduced_solutions_slab["time"])[:, -1],
#         [np.linalg.norm(dual_rhs_no_bc[i]) for i in range(len(dual_rhs_no_bc))], color='r', label="dual_rhs_no_bc")
# plt.grid()
# plt.legend()
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
# plt.xlabel('$t [$s$]$')
# plt.ylabel("norm")
# plt.xlim([0, slab_properties["n_total"]*time_step_size])
# plt.show()


# pointwise displacement
# plt.plot(np.arange(0, slab_properties["n_total"]*n_dofs["solperstep"]+1),
#          tuple(primal_solutions["value"][i][index2measuredisp] for i in range(slab_properties["n_total"]*n_dofs["solperstep"]+1)), label="pw displacement -fom")
# plt.plot(np.arange(0, slab_properties["n_total"]*n_dofs["solperstep"]+1),
#          tuple(projected_reduced_solutions["value"][i][index2measuredisp] for i in range(n_dofs["solperstep"]*slab_properties["n_total"]+1)), label="pw displacement -rom")
# plt.legend()
# plt.show()


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
# plt.xlim([0, slab_properties["n_total"]*time_step_size])
plt.savefig('images/cost_functional.png')

plt.show()


# plot temporal evolution of error and error estimate
plt.rcParams["figure.figsize"] = (10, 6)
plt.plot(np.vstack(projected_reduced_solutions_slab["time"])[:, -1],
         (np.array(temporal_interval_error)), c='#1f77b4', label="estimate")
plt.plot(np.vstack(projected_reduced_solutions_slab["time"])[:, -1],
         (J_h_t-J_r_t), color='r', label="error")
plt.grid()
plt.legend()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.xlabel('$t [$s$]$')
plt.ylabel("$error$")
# plt.xlim([0, slab_properties["n_total"]*time_step_size])
plt.savefig('images/estimate_vs_true_error.png')

plt.show()
