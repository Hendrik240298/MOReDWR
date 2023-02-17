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
INTERPOLATION_TYPE = "cubic"  # "linear", "cubic"
CASE = ""  # "two" or "moving"
MOTHER_PATH = "/home/ifam/fischer/Code/MORe_DWR/Heat/"
MOTHER_PATH = "/home/hendrik/Code/MORe_DWR/Heat/"
OUTPUT_PATH = MOTHER_PATH + "Data/2D/rotating_circle/slabwise/FOM/"
cycle = "cycle=4"
SAVE_PATH = MOTHER_PATH + "Data/2D/rotating_circle/slabwise/ROM/" + cycle + "/"
# SAVE_PATH = cycle + "/output_ROM/"

#"../../FOM/slabwise/output_" + CASE + "/dim=1/"

ENERGY_PRIMAL = 0.99999999
ENERGY_DUAL =   0.99999999


if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
if not os.path.exists(SAVE_PATH + "movie/"):    
    os.makedirs(SAVE_PATH + "movie/")


# %% load data
print(f"\n{'-'*12}\n| {cycle}: |\n{'-'*12}\n")
# NO BC
[data, row, column] = np.loadtxt(OUTPUT_PATH + cycle + "/matrix_no_bc.txt")
matrix_no_bc = scipy.sparse.csr_matrix(
    (data, (row.astype(int), column.astype(int))))

# matrix_no_bc_coo = coo_matrix((data,(row,column)),shape=(8450,8450))
# A=matrix_no_bc_coo[1,1]

[data, row, column] = np.loadtxt(
    OUTPUT_PATH + cycle + "/jump_matrix_no_bc.txt")
jump_matrix_no_bc = scipy.sparse.csr_matrix(
    (data, (row.astype(int), column.astype(int))))

[data, row, column] = np.loadtxt(OUTPUT_PATH + cycle + "/mass_matrix_no_bc.txt")
mass_matrix_no_bc = scipy.sparse.csr_matrix(
    (data, (row.astype(int), column.astype(int))))

rhs_no_bc = []
for f in sorted([f for f in os.listdir(OUTPUT_PATH + cycle)
                if "dual" not in f and "rhs_no_bc" in f]):
    rhs_no_bc.append(np.loadtxt(OUTPUT_PATH + cycle + "/" + f))

dual_rhs_no_bc = []
for f in sorted([f for f in os.listdir(
        OUTPUT_PATH + cycle) if "dual_rhs_no_bc" in f]):
    dual_rhs_no_bc.append(np.loadtxt(OUTPUT_PATH + cycle + "/" + f))

boundary_ids = np.loadtxt(OUTPUT_PATH + cycle +
                          "/boundary_id.txt").astype(int)

# %% applying BC to primal matrix
primal_matrix = matrix_no_bc.tocsr()
for row in boundary_ids:
    for col in primal_matrix.getrow(row).nonzero()[1]:
        primal_matrix[row, col] = 1. if row == col else 0.
        # for in_bc in range(len(rhs_no_bc)):
        #     if row == col: 
        #         rhs_no_bc[in_bc][col] = 1. 

# %% applying BC to dual matrix
dual_matrix_no_bc = matrix_no_bc.T.tocsr()
dual_matrix = matrix_no_bc.T.tocsr()
for row in boundary_ids:
    for col in dual_matrix.getrow(row).nonzero()[1]:
        dual_matrix[row, col] = 1. if row == col else 0.
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

# %% primal FOM solve
start_execution = time.time()
last_primal_solution = np.zeros_like(rhs_no_bc[0])
primal_solutions = []
for i in range(n_slabs):
    # creating primal rhs and applying BC to it
    primal_rhs = rhs_no_bc[i].copy()
    primal_rhs -= jump_matrix_no_bc.dot(last_primal_solution)
    for row in boundary_ids:
        primal_rhs[row] = 0.  # NOTE: hardcoding homogeneous Dirichlet BC
        
    primal_solutions.append(scipy.sparse.linalg.spsolve(primal_matrix, primal_rhs))
    last_primal_solution = primal_solutions[-1]
    
end_execution = time.time()
execution_time_FOM = end_execution - start_execution
# plot primal solution
primal_solution = np.hstack(primal_solutions)

# %% dual solve
last_dual_solution = np.zeros_like(dual_rhs_no_bc[0])
dual_solutions = []
for i in list(range(n_slabs))[::-1]:
    # creating dual rhs and applying BC to it
    dual_rhs = 2*mass_matrix_no_bc.dot(primal_solutions[i])
    # dual_rhs = dual_rhs_no_bc[i].copy()
    dual_rhs -= jump_matrix_no_bc.T.dot(last_dual_solution)
    for row in boundary_ids:
        dual_rhs[row] = 0.

    dual_solutions.append(
        scipy.sparse.linalg.spsolve(dual_matrix, dual_rhs))

    last_dual_solution = dual_solutions[-1]

# dual solution
dual_solutions = dual_solutions[::-1]


# %% goal functionals
J = {"u_h": 0., "u_r": 0.}
J_h_t = np.empty([n_slabs, 1])
for i in range(n_slabs):
    J_h_t[i] = np.dot(primal_solutions[i], mass_matrix_no_bc.dot(primal_solutions[i]))
J["u_h"] = np.sum(J_h_t)
    
# %%
time_dofs_per_time_interval = int(n_dofs["time"] / n_slabs)
# dofs_per_time_interval = time_dofs_per_time_interval * n_dofs["space"]

# %% initilaize ROM framework
total_energy = 0
pod_basis = np.empty([0, 0])
bunch = np.empty([0, 0])
singular_values = np.empty([0, 0])

# only use first solution of slab since we assume that solutions are quite similar
for slab_step in range(int(primal_solutions[0].shape[0] / n_dofs["space"])):
    print(slab_step)
    pod_basis, bunch, singular_values, total_energy = iPOD(pod_basis, 
                                                           bunch, 
                                                           singular_values, 
                                                           primal_solutions[0][range(slab_step * n_dofs["space"], (slab_step + 1) * n_dofs["space"])],
                                                           total_energy,
                                                           ENERGY_PRIMAL)

# change from the FOM to the POD basis
space_time_pod_basis = scipy.sparse.block_diag(
    [pod_basis] * time_dofs_per_time_interval)

print(pod_basis.shape)

reduced_system_matrix = reduce_matrix(matrix_no_bc,pod_basis,pod_basis)
reduced_jump_matrix = reduce_matrix(jump_matrix_no_bc,pod_basis,pod_basis)

# reduced_system_matrix = space_time_pod_basis.T.dot(
#     matrix_no_bc.dot(space_time_pod_basis)).toarray()
# reduced_jump_matrix = space_time_pod_basis.T.dot(
#     jump_matrix_no_bc.dot(space_time_pod_basis)).toarray()

# %% initilaize dual ROM framework 
total_energy_dual = 0
pod_basis_dual = np.empty([0, 0])
bunch_dual = np.empty([0, 0])
singular_values_dual = np.empty([0, 0])

#print(space_time_pod_basis.shape)

# onyl use first solution of slab since we assume that solutions are quite similar
for slab_step in range(int(dual_solutions[0].shape[0] / n_dofs["space"])):
    pod_basis_dual, bunch_dual, singular_values_dual, total_energy_dual = iPOD(pod_basis_dual, 
                                                                               bunch_dual, 
                                                                               singular_values_dual, 
                                                                               dual_solutions[0][range(slab_step * n_dofs["space"], (slab_step + 1) * n_dofs["space"])], 
                                                                               total_energy_dual,
                                                                               ENERGY_DUAL)

# change from the FOM to the POD basis
# space_time_pod_basis_dual = scipy.sparse.block_diag(
#     [pod_basis_dual] * time_dofs_per_time_interval)


reduced_dual_matrix = reduce_matrix(dual_matrix_no_bc,pod_basis_dual,pod_basis_dual)
reduced_dual_jump_matrix_no_bc = reduce_matrix(jump_matrix_no_bc,pod_basis_dual,pod_basis_dual)

reduced_mass_matrix_no_bc = reduce_matrix(mass_matrix_no_bc,pod_basis_dual,pod_basis)

reduced_matrix_no_bc_estimator = reduce_matrix(matrix_no_bc, pod_basis_dual, pod_basis)
reduced_jump_matrix_no_bc_estimator = reduce_matrix(jump_matrix_no_bc, pod_basis_dual, pod_basis)
reduced_mass_matrix_no_bc_cst_fct = reduce_matrix(mass_matrix_no_bc, pod_basis, pod_basis)
# reduced_dual_matrix = space_time_pod_basis_dual.T.dot(
#     dual_matrix_no_bc.dot(space_time_pod_basis_dual)).toarray()
#  reduced_dual_jump_matrix_no_bc = space_time_pod_basis_dual.T.dot(
#     jump_matrix_no_bc.dot(space_time_pod_basis_dual)).toarray()


# reduced_mass_matrix_no_bc = space_time_pod_basis_dual.T.dot(
#     mass_matrix_no_bc.dot(space_time_pod_basis)).toarray()

# %% primal ROM solve

LU_primal, piv_primal = scipy.linalg.lu_factor(reduced_system_matrix)
LU_dual, piv_dual     = scipy.linalg.lu_factor(reduced_dual_matrix)

reduced_solutions = []
# reduced_solutions_old = space_time_pod_basis.T.dot(np.zeros_like(primal_solutions[0]))
reduced_solutions_old = reduce_vector(np.zeros_like(primal_solutions[0]), pod_basis)


reduced_dual_solutions = []
# reduced_dual_solutions_old = space_time_pod_basis_dual.T.dot(np.zeros_like(dual_solutions[0]))
reduced_dual_solutions_old = reduce_vector(np.zeros_like(dual_solutions[0]), pod_basis)

projected_reduced_solutions = []
projected_reduced_solutions_before_enrichment = []
projected_reduced_dual_solutions = []

dual_residual = []
dual_residual.append(0)

temporal_interval_error = []
temporal_interval_error_relative = []
temporal_interval_error_incidactor = []

tol = 1e-10
tol_rel = 1e-2
tol_dual = 5e-1
forwardsteps = 10

# print("tol =     " + str(tol))
print("tol_rel       = " + str(tol_rel))
print("tol           = " + str(tol))
print(f"forward steps = {forwardsteps}")
print(" ")
extime_solve = 0.0
extime_dual_solve = 0.0
extime_error = 0.0
extime_update  =0.0

nb_buckets = 32 #64*2*2# 32 # int(2*slab_properties["n_total"]/len_block_evaluation)
len_block_evaluation = int(n_slabs/nb_buckets)

start_execution = time.time()

projected_reduced_solutions_buckets_combined = []
reduced_solutions_buckets_combined = []
temporal_interval_error_incidactor_combinded = []
temporal_interval_error_combinded = []
temporal_interval_error_relative_combinded = []
index_primal = 0
last_bucket_end_solution = np.zeros(matrix_no_bc.shape[0])
for it_bucket in range(nb_buckets):
    print("bucket " + str(it_bucket+1) + " of " + str(nb_buckets) + " of length: " + str(len_block_evaluation)) 
    bucket_shift = it_bucket*len_block_evaluation
    temporal_interval_error_incidactor = np.zeros(len_block_evaluation)
    
    while True:
        #primal ROM solve
        extime_solve_start = time.time()
        primal_reduced_solutions = [reduce_vector(last_bucket_end_solution, pod_basis)]
        for i in range(len_block_evaluation):                
            reduced_rhs = reduce_vector(rhs_no_bc[i+bucket_shift], pod_basis)
            reduced_rhs -= reduced_jump_matrix.dot(primal_reduced_solutions[-1])
            primal_reduced_solutions.append(scipy.linalg.lu_solve((LU_primal, piv_primal), reduced_rhs))         
        primal_reduced_solutions = primal_reduced_solutions[1:]
        extime_solve += time.time() - extime_solve_start
        
        # reversed forward dual solve
        extime_dual_solve_start = time.time()
        reduced_dual_solutions = [np.zeros(reduced_dual_jump_matrix_no_bc.shape[1])]
        for i in range(len_block_evaluation):
            reduced_dual_rhs = 2*reduced_mass_matrix_no_bc.dot(primal_reduced_solutions[-i]) #len(forwarded_reduced_solutions)-forwardstep])
            reduced_dual_rhs -= reduced_dual_jump_matrix_no_bc.T.dot(reduced_dual_solutions[-1])
            reduced_dual_solutions.append(scipy.linalg.lu_solve((LU_dual, piv_dual), reduced_dual_rhs)) 
        reduced_dual_solutions = reduced_dual_solutions[1:]
        reduced_dual_solutions = reduced_dual_solutions[::-1]
        extime_dual_solve += time.time() - extime_dual_solve_start
        
        # # project primal solution up
        # extime_project_start = time.time()
        # projected_reduced_solutions = []
        # for i in range(len_block_evaluation):
        #     projected_reduced_solutions.append(project_vector(primal_reduced_solutions[i], pod_basis))

        # # project dual solution up
        # projected_reduced_dual_solutions = []
        # for i in range(len_block_evaluation):
        #     projected_reduced_dual_solutions.append(project_vector(reduced_dual_solutions[i], pod_basis_dual))
        # extime_project += time.time() - extime_project_start
        
        extime_error_start = time.time()
        temporal_interval_error = []
        temporal_interval_error_relative = []
           
           
        for i in range(len_block_evaluation):
            tmp = -reduced_matrix_no_bc_estimator.dot(primal_reduced_solutions[i]) + reduce_vector(rhs_no_bc[i+bucket_shift],pod_basis_dual)
            if i > 0:
                tmp -= reduced_jump_matrix_no_bc_estimator.dot(primal_reduced_solutions[i - 1])
            else:
                tmp -= reduced_jump_matrix_no_bc_estimator.dot(reduce_vector(last_bucket_end_solution, pod_basis)) #reduce_vector(jump_matrix_no_bc.dot(last_bucket_end_solution),pod_basis_dual) #
            temporal_interval_error.append(np.dot(reduced_dual_solutions[i], tmp))
            goal_functional = np.dot(primal_reduced_solutions[i],  reduced_mass_matrix_no_bc_cst_fct.dot(primal_reduced_solutions[i]))
            temporal_interval_error_relative.append(np.abs(temporal_interval_error[-1])/np.abs(goal_functional+temporal_interval_error[-1]))
        extime_error += time.time() - extime_error_start
                 
        # temporal_interval_error_relative = temporal_interval_error
        estimated_error = np.max(np.abs(temporal_interval_error_relative))

        print(estimated_error)
        if estimated_error < tol_rel:
        # if estimated_error < tol:
            break
        else:
            extime_update_start = time.time()
            index_primal = np.argmax(np.abs(temporal_interval_error_relative))
            index_dual = index_primal
            print(str(index_primal) + ":   " + str(np.abs(temporal_interval_error_relative[index_primal])))
            print(" ")

            temporal_interval_error_incidactor[index_primal] = 1
            print(f"i:            {i}")
            print(f"index_primal: {index_primal}")
            if index_primal > 0:    
                old_projected_solution = project_vector(primal_reduced_solutions[index_primal-1], pod_basis)  # projected_reduced_solutions[index_primal - 1]
            else:
                old_projected_solution = last_bucket_end_solution
            
            pod_basis, reduced_system_matrix, reduced_jump_matrix, new_projection_solution, singular_values, total_energy = ROM_update(
                        pod_basis, 
                        # space_time_pod_basis, 
                        reduced_system_matrix, 
                        reduced_jump_matrix, 
                        old_projected_solution, 
                        rhs_no_bc[index_primal+bucket_shift].copy(),
                        jump_matrix_no_bc,
                        boundary_ids,
                        primal_matrix,
                        singular_values,
                        total_energy,
                        n_dofs,
                        time_dofs_per_time_interval,
                        matrix_no_bc,
                        ENERGY_PRIMAL)
            LU_primal, piv_primal = scipy.linalg.lu_factor(reduced_system_matrix)

            reduced_mass_matrix_no_bc = reduce_matrix(mass_matrix_no_bc, pod_basis_dual, pod_basis)
            
            forwarded_reduced_solutions = []
            forwarded_reduced_solutions.append(reduce_vector(new_projection_solution,pod_basis))

            for forwardstep in range(forwardsteps):
                if index_primal+forwardstep+1 +bucket_shift >= n_slabs:
                    break
                forwarded_reduced_rhs =  reduce_vector(rhs_no_bc[index_primal+forwardstep+1+bucket_shift],pod_basis)
                forwarded_reduced_rhs -= reduced_jump_matrix.dot(forwarded_reduced_solutions[-1])
                forwarded_reduced_solutions.append(scipy.linalg.lu_solve((LU_primal, piv_primal), forwarded_reduced_rhs))
                
            # reversed forward dual solve
            forwarded_reduced_dual_solutions = []
            forwarded_reduced_dual_rhs = 2*reduced_mass_matrix_no_bc.dot(forwarded_reduced_solutions[-1])
            forwarded_reduced_dual_solutions.append(scipy.linalg.lu_solve((LU_dual, piv_dual), forwarded_reduced_dual_rhs))

            for forwardstep in range(2,len(forwarded_reduced_solutions)+1,1):
                forwarded_reduced_dual_rhs = 2*reduced_mass_matrix_no_bc.dot(forwarded_reduced_solutions[-forwardstep]) #len(forwarded_reduced_solutions)-forwardstep])
                forwarded_reduced_dual_rhs -= reduced_dual_jump_matrix_no_bc.T.dot(forwarded_reduced_dual_solutions[-1])
                forwarded_reduced_dual_solutions.append(scipy.linalg.lu_solve((LU_dual, piv_dual), forwarded_reduced_dual_rhs))

            if len(forwarded_reduced_dual_solutions) == 1:
                forwarded_reduced_dual_solutions.append(forwarded_reduced_dual_solutions[-1])
                forwarded_reduced_dual_solutions[-2] = np.zeros_like(forwarded_reduced_dual_solutions[-1])

            reduced_dual_solutions = forwarded_reduced_dual_solutions[-1]
            
            pod_basis_dual, reduced_dual_matrix, reduced_dual_jump_matrix_no_bc, _, singular_values_dual, total_energy_dual = ROM_update_dual(
                        pod_basis_dual, 
                        # space_time_pod_basis_dual, 
                        reduced_dual_matrix, 
                        reduced_dual_jump_matrix_no_bc, #reduced_dual_jump_matrix*0, 
                        # space_time_pod_basis_dual.dot(forwarded_reduced_dual_solutions[-2]), 
                        project_vector(forwarded_reduced_dual_solutions[-2],pod_basis_dual), 
                        # projected_reduced_dual_solutions[-2],
                        2*mass_matrix_no_bc.dot( new_projection_solution),
                        jump_matrix_no_bc,
                        boundary_ids,
                        dual_matrix,
                        singular_values_dual,
                        total_energy_dual,
                        n_dofs,
                        time_dofs_per_time_interval,
                        dual_matrix_no_bc,
                        ENERGY_DUAL)    
            reduced_mass_matrix_no_bc = reduce_matrix(mass_matrix_no_bc, pod_basis_dual, pod_basis)

            reduced_matrix_no_bc_estimator = reduce_matrix(matrix_no_bc, pod_basis_dual, pod_basis)
            reduced_jump_matrix_no_bc_estimator = reduce_matrix(jump_matrix_no_bc, pod_basis_dual, pod_basis)
            reduced_mass_matrix_no_bc_cst_fct = reduce_matrix(mass_matrix_no_bc, pod_basis, pod_basis)
            # Lu decompostion of reduced matrices
            LU_dual, piv_dual     = scipy.linalg.lu_factor(reduced_dual_matrix)
            extime_update += time.time() - extime_update_start
            
            last_bucket_end_solution = project_vector(reduce_vector(last_bucket_end_solution,pod_basis),pod_basis)

    index_primal = len_block_evaluation-1
    
    last_bucket_end_solution = project_vector(primal_reduced_solutions[-1],pod_basis)   #projected_reduced_solutions[-1]
    
    for i in range(len_block_evaluation):
    #     reduced_solutions_buckets_combined.append(primal_reduced_solutions[i])
        temporal_interval_error_incidactor_combinded.append(temporal_interval_error_incidactor[i])
    #     temporal_interval_error_combinded.append(temporal_interval_error[i])
    #     temporal_interval_error_relative_combinded.append(temporal_interval_error_relative[i])
    
end_execution = time.time()
execution_time_ROM = end_execution - start_execution

# %% ---------------------- VERIFICATION ------------------------------------------------------------


extime_solve_start = time.time()
primal_reduced_solutions = [reduce_vector(np.zeros(matrix_no_bc.shape[0]), pod_basis)]
for i in range(n_slabs):                
    reduced_rhs = reduce_vector(rhs_no_bc[i], pod_basis)
    reduced_rhs -= reduced_jump_matrix.dot(primal_reduced_solutions[-1])
    primal_reduced_solutions.append(scipy.linalg.lu_solve((LU_primal, piv_primal), reduced_rhs))         
primal_reduced_solutions = primal_reduced_solutions[1:]
extime_solve += time.time() - extime_solve_start

# reversed forward dual solve
extime_dual_solve_start = time.time()
reduced_dual_solutions = [np.zeros(reduced_dual_jump_matrix_no_bc.shape[1])]
for i in range(n_slabs):
    reduced_dual_rhs = 2*reduced_mass_matrix_no_bc.dot(primal_reduced_solutions[-i]) #len(forwarded_reduced_solutions)-forwardstep])
    reduced_dual_rhs -= reduced_dual_jump_matrix_no_bc.T.dot(reduced_dual_solutions[-1])
    reduced_dual_solutions.append(scipy.linalg.lu_solve((LU_dual, piv_dual), reduced_dual_rhs)) 
reduced_dual_solutions = reduced_dual_solutions[1:]
reduced_dual_solutions = reduced_dual_solutions[::-1]
extime_dual_solve += time.time() - extime_dual_solve_start


extime_error_start = time.time()
temporal_interval_error = []
temporal_interval_error_relative = []
    
for i in range(n_slabs):
    tmp = -reduced_matrix_no_bc_estimator.dot(primal_reduced_solutions[i]) + reduce_vector(rhs_no_bc[i],pod_basis_dual)
    if i > 0:
        tmp -= reduced_jump_matrix_no_bc_estimator.dot(primal_reduced_solutions[i - 1])
    else:
        tmp -= reduced_jump_matrix_no_bc_estimator.dot(reduce_vector(np.zeros(matrix_no_bc.shape[0]), pod_basis)) 
    temporal_interval_error.append(np.dot(reduced_dual_solutions[i], tmp))
    goal_functional = np.dot(primal_reduced_solutions[i],  reduced_mass_matrix_no_bc_cst_fct.dot(primal_reduced_solutions[i]))
    temporal_interval_error_relative.append(np.abs(temporal_interval_error[-1])/np.abs(goal_functional+temporal_interval_error[-1]))
extime_error += time.time() - extime_error_start

estimated_error = np.max(np.abs(temporal_interval_error_relative))

print(f"Largest Error in Verification: {estimated_error}")


# %%
# reduced_solutions = reduced_solutions_buckets_combined
temporal_interval_error_incidactor = temporal_interval_error_incidactor_combinded
# temporal_interval_error = temporal_interval_error_combinded
# temporal_interval_error_relative = temporal_interval_error_relative_combinded


print("FOM time:         " + str(execution_time_FOM))
print("ROM time:         " + str(execution_time_ROM))
print("speedup: act/max: " + str(execution_time_FOM/execution_time_ROM) + " / " + str(len(temporal_interval_error_incidactor)/(2*np.sum(temporal_interval_error_incidactor))))
print("Size ROM:         " + str(pod_basis.shape[1]))
print("Size ROM - dual:  " + str(pod_basis_dual.shape[1]))
print("FOM solves:       " + str(np.sum(temporal_interval_error_incidactor)
                           ) + " / " + str(len(temporal_interval_error_incidactor)))
print(" ")
print("ROM Solve time:      " + str(extime_solve))
print("ROM dual Solve time: " + str(extime_dual_solve))
# print("Project time:        " + str(extime_project))
print("Error est time:      " + str(extime_error))
print("Update time:         " + str(extime_update))
print("Overall time:        " + str(extime_solve+extime_error+extime_update+extime_dual_solve))
print(" ")

original_stdout = sys.stdout  # Save a reference to the original standard output
# with open('output/speedup_' + CASE + '_cycle_' + str(cycle) + '.txt', 'a') as f:
#     sys.stdout = f # Change the standard output to the file we created.
#     print(str(execution_time_FOM) + ', ' + str(execution_time_ROM) + ', ' + str(execution_time_FOM/execution_time_ROM))
#     sys.stdout = original_stdout # Reset the standard output to its original value



J_r = 0.
J_r_t = np.empty([n_slabs, 1])
J_r_t_before_enrichement = np.empty([n_slabs, 1])
for i in range(n_slabs):
    # J_r_t[i] = np.dot(projected_reduced_solutions[i], mass_matrix_no_bc.dot(projected_reduced_solutions[i]))
    J_r_t[i] = np.dot(primal_reduced_solutions[i],  reduced_mass_matrix_no_bc_cst_fct.dot(primal_reduced_solutions[i]))
    # J_r_t_before_enrichement[i] = np.dot(projected_reduced_solutions_before_enrichment[i], mass_matrix_no_bc.dot(projected_reduced_solutions_before_enrichment[i]))
    # print(np.mean(projected_reduced_solutions[i] -projected_reduced_solutions_before_enrichment[i]))
J["u_r"] = np.sum(J_r_t) 
J_r_t_before_enrichement = J_r_t
print("J(u_h) =", J["u_h"])
# TODO: in the future compare J(u_r) for different values of rprojected_reduced_dual_solutions
print("J(u_r) =", J["u_r"])
print(" ")

# %% real error in cost functional
real_temporal_interval_error_relative = []
for i in range(1,n_slabs,1):
    real_temporal_interval_error_relative.append(np.abs((J_h_t[i]-J_r_t_before_enrichement[i])/J_h_t[i]))

# %% error estimation
true_error = J['u_h'] - J['u_r']

real_max_error = np.max(np.abs((J_h_t - J_r_t)/J_h_t))
real_max_error_index = np.argmax(np.abs((J_h_t - J_r_t)/J_h_t))
estimated_max_error = np.max(np.abs(temporal_interval_error_relative))
estimated_max_error_index = np.argmax(np.abs(temporal_interval_error_relative))

print(f"Largest estimated error at: {estimated_max_error_index} with: {estimated_max_error}")
print(f"Largest real error at:      {real_max_error_index} with: {real_max_error}")
print(f"We instead estimated:                 {np.abs(temporal_interval_error_relative)[real_max_error_index]}")


#print(np.sum(error_tol.astype(int)))

np.abs((J_h_t - J_r_t)/J_h_t)

J_r_t = J_r_t_before_enrichement

true_tol = np.abs((J_h_t - J_r_t)/J_h_t) > tol_rel
esti_tol = np.abs(temporal_interval_error_relative) > tol_rel

#esti_tol = np.abs(np.array(temporal_interval_error).reshape(-1,1)/J_h_t) > tol_rel


if np.sum(true_tol) == np.sum(esti_tol):
    print("estimator works perfectly")
else:
    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(true_tol.astype(int), esti_tol.astype(int))
    eltl, egtl, eltg, egtg = confusion_matrix.ravel()
    # n_slabs=100

    print(f"(error > tol & esti < tol): {eltg} ({round(100 * eltg / n_slabs,1)} %)  (very bad)")
    print(f"(error < tol & esti > tol): {egtl} ({round(100 * egtl / n_slabs,1)} %)  (bad)")
    print(f"(error > tol & esti > tol): {egtg} ({round(100 * egtg / n_slabs,1)} %)  (good)")
    print(f"(error < tol & esti < tol): {eltl} ({round(100 * eltl / n_slabs,1)} %)  (good)")



n_slabs_filter=0
eff_alternative_1 = 0
eff_alternative_2 = 0
#vft = np.zeros((2,2))
#confusion_matrix = {"EGTL": 0, "EGTG": 0, "ELTG": 0, "ELTL": 0}
predicted_tol = ((J_h_t-J_r_t) > tol).astype(int)
# true_tol = 
for i in range(1,n_slabs,1):
    n_slabs_filter += -(temporal_interval_error_incidactor[i]-1) 
    eff_alternative_1 += (J_h_t[i]-J_r_t[i])/temporal_interval_error[i]
    eff_alternative_2 += -(temporal_interval_error_incidactor[i]-1)*np.abs((J_h_t[i]-J_r_t[i])/temporal_interval_error[i]) # filter only non updated
     
eff_alternative_1 /= (2*n_slabs)
eff_alternative_2 /= (2*n_slabs_filter)
# using FOM dual solution
print("\nUsing z_h:")
print("----------")
error_estimator = sum(temporal_interval_error)
print(f"True error:          {true_error}")
print(f"Estimated error:     {error_estimator}")
print(f"Effectivity index 1: {abs(true_error / error_estimator)}")
print(f"Effectivity index 2: {eff_alternative_1[0]}")
print(f"Effectivity index 3: {eff_alternative_2[0]}")


# %% Ploting
# Plot 3: temporal error
# WARNING: hardcoding end time T = 4.
time_step_size = 10.0 / (n_dofs["time"] / 2)
xx, yy = [], []
xx_FOM, yy_FOM = [], []
cc = []
# for i, error in enumerate(temporal_interval_error):
# for i, error in enumerate(temporal_interval_error_relative):
for i, error in enumerate(real_temporal_interval_error_relative):

    if temporal_interval_error_incidactor[i] == 0:
        xx += [i * time_step_size,
               (i + 1) * time_step_size, (i + 1) * time_step_size]
        yy += [abs(error), abs(error), np.inf]
    else:
        xx_FOM += [i * time_step_size,
                   (i + 1) * time_step_size, (i + 1) * time_step_size]
        yy_FOM += [abs(error), abs(error), np.inf]
    #     cc += ['g']
    # axs[2].plot(xx, yy)
    # axs[2].plot(xx_FOM, yy_FOM, 'r')
    # axs[2].set_xlabel("$t$")
    # axs[2].set_ylabel("$\\eta$")
    # axs[2].set_yscale("log")
    # axs[2].set_title("temporal error estimate")

# plot temporal error
plt.rc('text', usetex=True)
# plt.rcParams["figure.figsize"] = (10,2)
plt.plot(xx, yy, label="ROM solves")
plt.plot(xx_FOM, yy_FOM, color='r', label="FOM solves")
plt.plot([0,10],[1e-2,1e-2], '--', color='green') #, label="1\% relative error")
plt.text(7.5, 1.2e-2, "$1\%$ relative error" , fontsize=12, color='green')
plt.grid()
plt.legend( fontsize=13)
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlabel('$t \; [$s$]$',fontsize=15)
# plt.ylabel("$\eta_{\rel}\\raisebox{-.5ex}{$|$}_{Q_l}$")
# plt.ylabel("$\eta\\raisebox{-.5ex}{$|$}_{I_m}$",fontsize=16)
plt.ylabel("$^{|J(u^{FOM}) - J(u^{ROM})|}/_{|J(u^{FOM})|}$",fontsize=16)
plt.yscale("log")
plt.xlim([0, n_slabs*time_step_size])
#plt.title("temporal evaluation of cost funtional")
plt.savefig(SAVE_PATH + "temporal_error_cost_funtional.eps", format='eps')
plt.savefig(SAVE_PATH + "temporal_error_cost_funtional.png", format='png')

plt.show()



# Plot 4: local effectivity
# WARNING: hardcoding end time T = 10.
time_step_size = 10.0 / (n_dofs["time"] / 2)
xx, yy = [], []
xx_FOM, yy_FOM = [], []
xxe, yye = [], []
xxe_FOM, yye_FOM = [], []
cc = []
for i, error in enumerate(temporal_interval_error):
    if temporal_interval_error_incidactor[i] == 0:
        xx += [i * time_step_size,
                    (i + 1) * time_step_size, (i + 1) * time_step_size]
        yy += [abs(J_h_t[i]-J_r_t[i]),abs(J_h_t[i]-J_r_t[i]), np.inf]      
    else:
        xx_FOM += [i * time_step_size,
                    (i + 1) * time_step_size, (i + 1) * time_step_size]
        yy_FOM += [abs(J_h_t[i]-J_r_t[i]), abs(J_h_t[i]-J_r_t[i]), np.inf]
        
    if temporal_interval_error_incidactor[i] == 0:
        xxe += [i * time_step_size,
                    (i + 1) * time_step_size, (i + 1) * time_step_size]
        yye += [abs(error), abs(error), np.inf]      
    else:
        xxe_FOM += [i * time_step_size,
                    (i + 1) * time_step_size, (i + 1) * time_step_size]
        yye_FOM += [abs(error), abs(error), np.inf]
        
   # xx += [i * time_step_size,
   #             (i + 1) * time_step_size, (i + 1) * time_step_size]
   # yy += [abs(J_h_t[i]-J_r_t[i])/abs(error), abs(J_h_t[i]-J_r_t[i])/abs(error), np.inf]
   
    #     cc += ['g']
    # axs[2].plot(xx, yy)
    # axs[2].plot(xx_FOM, yy_FOM, 'r')
    # axs[2].set_xlabel("$t$")
    # axs[2].set_ylabel("$\\eta$")
    # axs[2].set_yscale("log")
    # axs[2].set_title("temporal error estimate")

# plot temporal error
plt.rc('text', usetex=True)
# plt.rcParams["figure.figsize"] = (10,2)
plt.plot(xx, yy, label="error", color='b')
plt.plot(xx_FOM, yy_FOM, color='b')
plt.plot(xxe, yye, label="estimate", color='r')
plt.plot(xxe_FOM, yye_FOM, color='r')
plt.grid()
plt.legend()
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlabel('$t \; [$s$]$')
plt.ylabel("$I_{eff}\\raisebox{-.5ex}{$|$}_{Q_l}$")
plt.yscale("log")
plt.xlim([0, n_slabs*time_step_size])

#plt.title("temporal evaluation of cost funtional")
plt.savefig(SAVE_PATH + "effectivity.eps", format='eps')
plt.savefig(SAVE_PATH + "effectivity.png", format='png')

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
plt.savefig(SAVE_PATH + "temporal_cost_funtional.eps", format='eps')
plt.savefig(SAVE_PATH + "temporal_cost_funtional.png", format='png')
plt.show()


# plot temporal evolution of cost funtiponal
plt.rcParams["figure.figsize"] = (10,2)
plt.plot(np.arange(0, n_slabs*time_step_size, time_step_size),
         temporal_interval_error, c='#1f77b4', label="estimate")
plt.plot(np.arange(0, n_slabs*time_step_size, time_step_size),
         J_h_t-J_r_t, color='r', label="error")
# plt.plot(np.arange(0, n_slabs*time_step_size, time_step_size),
#          J_h_t-J_r_t_before_enrichement, color='g', label="error_before_enrichment")
plt.grid()
plt.legend()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.xlabel('$t \; [$s$]$')
plt.ylabel("$error$")
plt.xlim([0, n_slabs*time_step_size])
#plt.title("temporal evaluation of cost funtional")
plt.savefig(SAVE_PATH + "error_estimate_over_time.eps", format='eps')
plt.savefig(SAVE_PATH + "error_estimate_over_time.png", format='png')
plt.show()


