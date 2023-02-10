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

# ENERGY_DUAL   = {"displacement": 1-1e-8,
#                  "velocity":     1-1e-8}
# ENERGY_PRIMAL = {"displacement": 1-1e-6,
#                  "velocity":     1-1e-6}

ENERGY_PRIMAL = {"displacement": 0.99999999999,
                 "velocity":     0.99999999999}
ENERGY_DUAL = {"displacement": 0.99999999999,
                 "velocity":     0.99999999999}

# ENERGY_PRIMAL = {"displacement": 0.999999,
#                  "velocity":     0.999999}
# ENERGY_DUAL = {"displacement":   0.999999,
#                  "velocity":     0.999999}

# %% read in properties connected to discretization
n_dofs, slab_properties, index2measuredisp, dof_matrix, grid = read_in_discretization(
    OUTPUT_PATH + cycle)

slab_properties["n_time_unknowns"] += 1 # since we use whole block system
# print("n_time_unknowns", slab_properties["n_time_unknowns"])
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
matrix_no_bc= matrix_no_bc + mass_matrix_no_bc

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


# slab_properties["n_total"] = 100

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


for i in list(range(slab_properties["n_total"]-10,slab_properties["n_total"]))[::-1]: #list(range(slab_properties["n_total"]))[::-1]:
    if i == len(list(range(slab_properties["n_total"])))-1:
        print("I was here")
        dual_rhs = dual_system_rhs[i].copy() + mass_matrix_down_left.dot(np.zeros(dual_matrix.shape[0]))
    else:
        dual_rhs = dual_system_rhs[i].copy() + mass_matrix_down_left.dot(dual_solutions_slab["value"][-1])
    dual_solution =  scipy.sparse.linalg.spsolve(dual_matrix, dual_rhs)

    dual_solutions_slab["value"].append(dual_solution)
    dual_solutions_slab["time"].append(slab_properties["time_points"][i])


dual_solutions_slab["value"] = dual_solutions_slab["value"][::-1]
dual_solutions_slab["time"] = dual_solutions_slab["time"][::-1]

end_execution = time.time()
execution_time_dual_FOM = end_execution - start_execution
print("Dual FOM time:   " + str(execution_time_dual_FOM))


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

bunch_size = 1 # 1

total_energy = {"displacement": 0, "velocity": 0}
pod_basis = {"displacement": np.empty([0, 0]), "velocity": np.empty([0, 0])}
bunch = {"displacement": np.empty([0, 0]), "velocity": np.empty([0, 0])}
singular_values = {"displacement": np.empty([0, 0]), "velocity": np.empty([0, 0])}

# for primal_solution in primal_solutions["value"][1]:# [0:400]:
primal_solution = primal_solutions["value"][1]
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


total_energy_dual = {"displacement": 0, "velocity": 0}
pod_basis_dual = {"displacement": np.empty([0, 0]), "velocity": np.empty([0, 0])}
bunch_dual = {"displacement": np.empty([0, 0]), "velocity": np.empty([0, 0])}
singular_values_dual = {"displacement": np.empty([0, 0]), "velocity": np.empty([0, 0])}

# for dual_solution in dual_solutions["value"][1]:# [0:400]:
dual_solution = dual_solutions["value"][1]
pod_basis_dual["displacement"], bunch_dual["displacement"], singular_values_dual["displacement"], total_energy_dual["displacement"] \
    = iPOD(pod_basis_dual["displacement"],
            bunch_dual["displacement"],
            singular_values_dual["displacement"],
            dual_solution[0:n_dofs["space"]],
            total_energy_dual["displacement"],
            ENERGY_DUAL["displacement"],
            bunch_size)
pod_basis_dual["velocity"], bunch_dual["velocity"], singular_values_dual["velocity"], total_energy_dual["velocity"] \
    = iPOD(pod_basis_dual["velocity"],
            bunch_dual["velocity"],
            singular_values_dual["velocity"],
            dual_solution[n_dofs["space"]:2 * n_dofs["space"]],
            total_energy_dual["velocity"],
            ENERGY_DUAL["velocity"],
            bunch_size)


print(pod_basis["displacement"].shape[1])
print(pod_basis["velocity"].shape[1])


# * reduced matrices
# primal
mass_matrix_up_right_reduced = reduce_matrix(mass_matrix_up_right_no_bc, pod_basis, pod_basis)
system_matrix_reduced = reduce_matrix(matrix_no_bc, pod_basis, pod_basis)

#dual
mass_matrix_down_left_reduced = reduce_matrix(mass_matrix_down_left_no_bc, pod_basis_dual, pod_basis_dual)
dual_matrix_reduced = reduce_matrix(dual_matrix_no_bc, pod_basis_dual, pod_basis_dual)
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
tol = 1e-5
tol_rel = 1e-2
tol_dual = 5e-1
forwardsteps = 5

# print("tol =     " + str(tol))
print("tol_rel       = " + str(tol_rel))
print("tol           = " + str(tol))
print(f"forward steps = {forwardsteps}")
print(" ")
extime_rom_solve = 0.0
extime_projection = 0.0
extime_estimate = 0.0
extime_update = 0.0

n_dofs["reduced_primal"] = pod_basis["displacement"].shape[1] + \
    pod_basis["velocity"].shape[1]

# alpha = 1.1
projected_reduced_solutions_slab = {"value": [], "time": []}

# bunch_size = 1# slab_properties["n_time_unknowns"]

start_execution_whole = time.time()

    


nb_buckets = 10 # int(2*slab_properties["n_total"]/len_block_evaluation)
len_block_evaluation = int(slab_properties["n_total"]/nb_buckets)


projected_reduced_solutions_slab_buckets_combined = {"value": [], "time": []}
temporal_interval_error_incidactor_combinded = []
temporal_interval_error_combinded = []
temporal_interval_error_relative_combinded = []

last_bucket_end_solution = np.zeros(matrix_no_bc.shape[0])
for it_bucket in range(nb_buckets):
    print("bucket " + str(it_bucket+1) + " of " + str(nb_buckets) + " of length: " + str(len_block_evaluation)) 
    bucket_shift = it_bucket*len_block_evaluation
    temporal_interval_error_incidactor = np.zeros(len_block_evaluation)
    while(True):
        start_execution = time.time()
        primal_reduced_solutions = [reduce_vector(last_bucket_end_solution, pod_basis)] #[np.zeros(system_matrix_reduced.shape[0])]
        
        # solve reduced primal system
        for i in range(len_block_evaluation):
            primal_rhs_reduced = reduce_vector(primal_system_rhs[i+bucket_shift].copy(), pod_basis) + mass_matrix_up_right_reduced.dot(primal_reduced_solutions[-1])
            primal_reduced_solutions.append(np.linalg.solve(system_matrix_reduced, primal_rhs_reduced))
        primal_reduced_solutions = primal_reduced_solutions[1:]
        
        # solve reduced dual problem
        dual_reduced_solutions_slab = [np.zeros(mass_matrix_down_left_reduced.shape[0])] # not really reduced
        for i in range(len_block_evaluation):
            dual_rhs_reduced = reduce_vector(dual_system_rhs[0].copy(), pod_basis_dual)  + mass_matrix_down_left_reduced.dot(dual_reduced_solutions_slab[-1])
            dual_reduced_solutions_slab.append(np.linalg.solve(dual_matrix_reduced, dual_rhs_reduced))
        dual_reduced_solutions_slab = dual_reduced_solutions_slab[1:]
        dual_reduced_solutions_slab = dual_reduced_solutions_slab[::-1]
        extime_rom_solve += time.time() - start_execution

        start_execution = time.time()    
        # project primal solution up
        projected_reduced_solutions_slab = {"value": [], "time": []} 
        for i in range(len_block_evaluation):
            projected_reduced_solutions_slab["value"].append(project_vector(primal_reduced_solutions[i], pod_basis))
            projected_reduced_solutions_slab["time"].append(slab_properties["time_points"][i+bucket_shift])

        # project dual solution up
        projected_dual_reduced_solutions_slab = []
        for i in range(len_block_evaluation):
            projected_dual_reduced_solutions_slab.append(project_vector(dual_reduced_solutions_slab[i], pod_basis_dual))
        extime_projection += time.time() - start_execution
        
        start_execution = time.time()
        # compute estimator for each time step
        temporal_interval_error = []
        temporal_interval_error_relative = []
        goal_func_on_slab = []
        for i in range(len_block_evaluation):
            if i > 0:
                residual_slab = - matrix_no_bc.dot(projected_reduced_solutions_slab["value"][i]) + rhs_no_bc[i+bucket_shift].copy() + mass_matrix_up_right_no_bc.dot(projected_reduced_solutions_slab["value"][i-1])
            else:
                residual_slab = - matrix_no_bc.dot(projected_reduced_solutions_slab["value"][i]) + rhs_no_bc[i+bucket_shift].copy() + mass_matrix_up_right_no_bc.dot( last_bucket_end_solution)
            temporal_interval_error.append(projected_dual_reduced_solutions_slab[i].dot(residual_slab))
            goal_func_on_slab.append(projected_reduced_solutions_slab["value"][i].dot(dual_rhs_no_bc[i]))
            if np.abs(temporal_interval_error[i] + goal_func_on_slab[i]) > 1e-12:
                temporal_interval_error_relative.append(temporal_interval_error[i]/(temporal_interval_error[i] + goal_func_on_slab[i]))
            else:
                temporal_interval_error_relative.append(0)
        extime_estimate += time.time() - start_execution
        
        estimated_error = np.abs(np.sum(temporal_interval_error))
        # estimated_error = np.sum(np.abs(temporal_interval_error))
        estimated_error = np.max(np.abs(temporal_interval_error_relative))
        print(estimated_error)
        if estimated_error < tol_rel:
            break
        else:
            start_execution = time.time()
            
            # compute index whit largest primal error
            # index_primal = np.argmax(np.abs(temporal_interval_error))
            index_primal = np.argmax(np.abs(temporal_interval_error_relative))
            index_dual = index_primal
            print(str(index_primal) + ":   " + str(np.abs(temporal_interval_error_relative[index_primal])))
            print(" ")

            temporal_interval_error_incidactor[index_primal] = 1
            
            # solve primal FOM system
            if index_primal > 0:
                primal_rhs = primal_system_rhs[index_primal+bucket_shift].copy() + mass_matrix_up_right.dot(projected_reduced_solutions_slab["value"][index_primal-1])
            else:
                primal_rhs = primal_system_rhs[index_primal+bucket_shift].copy() + mass_matrix_up_right.dot(last_bucket_end_solution)
            recomputed_reduced_solutions_slab = scipy.sparse.linalg.spsolve(primal_matrix, primal_rhs)
            
            # solve dual FOM system
            if index_dual < len_block_evaluation-1:
                dual_rhs = dual_system_rhs[index_dual].copy() + mass_matrix_down_left.dot(projected_dual_reduced_solutions_slab[index_dual+1])
            else:
                dual_rhs = dual_system_rhs[index_dual].copy()
            recomputed_reduced_dual_solutions_slab =  scipy.sparse.linalg.spsolve(dual_matrix, dual_rhs)
        
            # update ROM basis
            for k in range(slab_properties["n_time_unknowns"]):
                primal_solution = recomputed_reduced_solutions_slab[k*n_dofs["time_step"]:(k+1)*n_dofs["time_step"]]
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
                    
            for k in range(slab_properties["n_time_unknowns"]):
                dual_solution = recomputed_reduced_dual_solutions_slab[k*n_dofs["time_step"]:(k+1)*n_dofs["time_step"]]            
                pod_basis_dual["displacement"], bunch_dual["displacement"], singular_values_dual["displacement"], total_energy_dual["displacement"] \
                    = iPOD(pod_basis_dual["displacement"],
                        bunch_dual["displacement"],
                        singular_values_dual["displacement"],
                        dual_solution[0:n_dofs["space"]],
                        total_energy_dual["displacement"],
                        ENERGY_DUAL["displacement"],
                        bunch_size)
                pod_basis_dual["velocity"], bunch_dual["velocity"], singular_values_dual["velocity"], total_energy_dual["velocity"] \
                    = iPOD(pod_basis_dual["velocity"],
                        bunch_dual["velocity"],
                        singular_values_dual["velocity"],
                        dual_solution[n_dofs["space"]:2 * n_dofs["space"]],
                        total_energy_dual["velocity"],
                        ENERGY_DUAL["velocity"],
                        bunch_size)

            # * reduced matrices
            mass_matrix_up_right_reduced = reduce_matrix(mass_matrix_up_right_no_bc, pod_basis, pod_basis)
            system_matrix_reduced = reduce_matrix(matrix_no_bc, pod_basis, pod_basis)
            mass_matrix_down_left_reduced = reduce_matrix(mass_matrix_down_left_no_bc, pod_basis_dual, pod_basis_dual)
            dual_matrix_reduced = reduce_matrix(dual_matrix_no_bc, pod_basis_dual, pod_basis_dual)
            extime_update += time.time() - start_execution
    
    
    
    index_primal = len_block_evaluation-1
    primal_rhs = primal_system_rhs[index_primal+bucket_shift].copy() + mass_matrix_up_right.dot(projected_reduced_solutions_slab["value"][index_primal-1])
    projected_reduced_solutions_slab["value"][-1] = scipy.sparse.linalg.spsolve(primal_matrix, primal_rhs)
    
    last_bucket_end_solution = projected_reduced_solutions_slab["value"][-1]
    # last_bucket_end_solution = primal_solutions_slab["value"][(it_bucket+1)*len_block_evaluation-1]
    
    # total_energy_dual = {"displacement": 0, "velocity": 0}
    # pod_basis_dual = {"displacement": np.empty([0, 0]), "velocity": np.empty([0, 0])}
    # bunch_dual = {"displacement": np.empty([0, 0]), "velocity": np.empty([0, 0])}
    # singular_values_dual = {"displacement": np.empty([0, 0]), "velocity": np.empty([0, 0])}

    # # for dual_solution in dual_solutions["value"][1]:# [0:400]:
    # dual_solution = dual_solutions["value"][1]
    # pod_basis_dual["displacement"], bunch_dual["displacement"], singular_values_dual["displacement"], total_energy_dual["displacement"] \
    #     = iPOD(pod_basis_dual["displacement"],
    #             bunch_dual["displacement"],
    #             singular_values_dual["displacement"],
    #             dual_solution[0:n_dofs["space"]],
    #             total_energy_dual["displacement"],
    #             ENERGY_DUAL["displacement"],
    #             bunch_size)
    # pod_basis_dual["velocity"], bunch_dual["velocity"], singular_values_dual["velocity"], total_energy_dual["velocity"] \
    #     = iPOD(pod_basis_dual["velocity"],
    #             bunch_dual["velocity"],
    #             singular_values_dual["velocity"],
    #             dual_solution[n_dofs["space"]:2 * n_dofs["space"]],
    #             total_energy_dual["velocity"],
    #             ENERGY_DUAL["velocity"],
    #             bunch_size)
    # mass_matrix_down_left_reduced = reduce_matrix(mass_matrix_down_left_no_bc, pod_basis_dual, pod_basis_dual)
    # dual_matrix_reduced = reduce_matrix(dual_matrix_no_bc, pod_basis_dual, pod_basis_dual)
    
    
    
    
    
    
    
    
    J_r_t = np.empty([len_block_evaluation, 1])
    J_h_t = np.empty([len_block_evaluation, 1])
    for i in range(len_block_evaluation):
        J_r_t[i] = projected_reduced_solutions_slab["value"][i].dot(dual_rhs_no_bc[i])
        J_h_t[i] = primal_solutions_slab["value"][i+bucket_shift].dot(dual_rhs_no_bc[i])

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
    
    print(" ")
    print(" ")
    
    for i in range(len_block_evaluation):
        projected_reduced_solutions_slab_buckets_combined["value"].append(projected_reduced_solutions_slab["value"][i])
        projected_reduced_solutions_slab_buckets_combined["time"].append(projected_reduced_solutions_slab["time"][i])
        temporal_interval_error_incidactor_combinded.append(temporal_interval_error_incidactor[i])
        temporal_interval_error_combinded.append(temporal_interval_error[i])
        temporal_interval_error_relative_combinded.append(temporal_interval_error_relative[i])
    
end_execution = time.time()
execution_time_ROM = end_execution - start_execution_whole

projected_reduced_solutions_slab["value"] = projected_reduced_solutions_slab_buckets_combined["value"]
projected_reduced_solutions_slab["time"] = projected_reduced_solutions_slab_buckets_combined["time"]
temporal_interval_error_incidactor = temporal_interval_error_incidactor_combinded
temporal_interval_error = temporal_interval_error_combinded
temporal_interval_error_relative = temporal_interval_error_relative_combinded

print("FOM time:             " + str(execution_time_FOM))
print("ROM time:             " + str(execution_time_ROM))
print("speedup: act/max:     " + str(execution_time_FOM/execution_time_ROM) + " / " + str(len(temporal_interval_error_incidactor)/(np.sum(temporal_interval_error_incidactor))))
print("Size ROM: u/v         " + str(pod_basis["displacement"].shape[1]) + " / " + str(pod_basis["velocity"].shape[1]))
print("Size ROM - dual: u/v  " + str(pod_basis_dual["displacement"].shape[1]) + " / " + str(pod_basis_dual["velocity"].shape[1]))
print("FOM solves:           " + str(np.sum(temporal_interval_error_incidactor)
                             ) + " / " + str(len(temporal_interval_error_incidactor)))
print("Ex times in algo:")


print("ROM :       " + str(extime_rom_solve))
print("Projection: " + str(extime_projection))
print("Estimate:   " + str(extime_estimate))
print("Update:     " + str(extime_update))
print(" ")
projected_reduced_solution = np.hstack(projected_reduced_solutions)

# %% Computing reduced and full Cost functional

J_r_t = np.empty([slab_properties["n_total"], 1])
J_h_t = np.empty([slab_properties["n_total"], 1])
for i in range(slab_properties["n_total"]):
    J_r_t[i] = projected_reduced_solutions_slab["value"][i].dot(dual_rhs_no_bc[i])
    J_h_t[i] = primal_solutions_slab["value"][i].dot(dual_rhs_no_bc[i])

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

# %% error metric

true_tol = np.abs((J_h_t - J_r_t)/J_h_t) > tol_rel
esti_tol = np.abs(temporal_interval_error_relative) > tol_rel

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(true_tol.astype(int), esti_tol.astype(int))
eltl, egtl, eltg, egtg = confusion_matrix.ravel()
# n_slabs=100

print(f"(error > tol & esti < tol): {eltg} ({round(100 * eltg / slab_properties['n_total'],1)} %)  (very bad)")
print(f"(error < tol & esti > tol): {egtl} ({round(100 * egtl / slab_properties['n_total'],1)} %)  (bad)")
print(f"(error > tol & esti > tol): {egtg} ({round(100 * egtg / slab_properties['n_total'],1)} %)  (good)")
print(f"(error < tol & esti < tol): {eltl} ({round(100 * eltl / slab_properties['n_total'],1)} %)  (good)")





# %% Plotting
time_step_size = 40.0 / (slab_properties["n_total"])



# Cost functional
plt.rcParams["figure.figsize"] = (10, 6)
plt.plot(np.vstack(projected_reduced_solutions_slab["time"])[:, -1],
         J_h_t, color='r', label="$u_h$")
plt.plot(np.vstack(projected_reduced_solutions_slab["time"])[:, -1],
         J_r_t, '--', c='#1f77b4', label="$u_N$")
plt.grid()
plt.legend()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.xlabel('$t [$s$]$')
plt.ylabel("$J(u)$")
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
plt.savefig('images/estimate_vs_true_error.png')
plt.show()


# plot temporal evolution of relative error and error estimate
plt.rcParams["figure.figsize"] = (10, 6)
plt.plot(np.vstack(projected_reduced_solutions_slab["time"])[:, -1],
         np.abs(np.array(temporal_interval_error_relative)), c='#1f77b4', label="estimate - relative")
plt.plot(np.vstack(projected_reduced_solutions_slab["time"])[:, -1],
         np.abs((J_h_t-J_r_t)/J_h_t), color='r', label="error - relative")
plt.plot([projected_reduced_solutions_slab["time"][0][-1], projected_reduced_solutions_slab["time"][-1][-1]],
         [tol_rel, tol_rel], color='y', label="tolerance - relative")
plt.grid()
plt.legend()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.xlabel('$t [$s$]$')
plt.ylabel("$error$")
plt.yscale('log')
plt.savefig('images/relative_estimate_vs_true_error.png')
plt.show()