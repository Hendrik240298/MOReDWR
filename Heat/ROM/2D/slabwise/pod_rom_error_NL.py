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
import imageio

PLOTTING = False
INTERPOLATION_TYPE = "cubic"  # "linear", "cubic"
CASE = ""  # "two" or "moving"
MOTHER_PATH = "/home/ifam/fischer/Code/MORe_DWR/Heat/"
OUTPUT_PATH = MOTHER_PATH + "Data/2D/rotating_circle/slabwise/FOM/"
cycle = "cycle=5"
SAVE_PATH = MOTHER_PATH + "Data/2D/rotating_circle/slabwise/ROM/" + cycle + "/"
# SAVE_PATH = cycle + "/output_ROM/"

#"../../FOM/slabwise/output_" + CASE + "/dim=1/"

ENERGY_PRIMAL = 0.999999
ENERGY_DUAL = 0.99999999

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
        
    primal_solutions.append(
        scipy.sparse.linalg.spsolve(primal_matrix, primal_rhs))
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

tol = 5e-4/(n_slabs)
tol_rel = 1e-2
tol_dual = 5e-1
forwardsteps = 20

# print("tol =     " + str(tol))
print("tol_rel       = " + str(tol_rel))
print("tol           = " + str(tol))
print(f"forward steps = {forwardsteps}")
print(" ")
start_execution = time.time()
extime_solve = 0.0
extime_dual_solve = 0.0
extime_error = 0.0
extime_update  =0.0


for i in range(n_slabs):
    start_time = time.time()
    
    #primal ROM solve
    # reduced_rhs = space_time_pod_basis.T.dot(rhs_no_bc[i])
    reduced_rhs = reduce_vector(rhs_no_bc[i], pod_basis)
    reduced_rhs -= reduced_jump_matrix.dot(reduced_solutions_old)
    # reduced_solutions = np.linalg.solve(reduced_system_matrix, reduced_rhs)
    reduced_solutions = scipy.linalg.lu_solve((LU_primal, piv_primal), reduced_rhs)
    # reduced_solutions = scipy.sparse.linalg.spsolve(
        # reduced_system_matrix, reduced_rhs)
    # projected_reduced_solutions.append(space_time_pod_basis.dot(reduced_solutions))
    projected_reduced_solutions.append(project_vector(reduced_solutions, pod_basis))
    projected_reduced_solutions_before_enrichment.append(projected_reduced_solutions[-1])
    extime_solve += time.time() - start_time
    
    start_time = time.time()
    # compute ADJOINT SOLUTION  
    # forward primal solve
    forwarded_reduced_solutions = []
    forwarded_reduced_solutions.append(reduced_solutions)
    # forwarded_reduced_solutions.append(space_time_pod_basis.T.dot(primal_solutions[i]))

    for forwardstep in range(forwardsteps):
        if i+forwardstep+1 >= n_slabs:
            break
        # forwarded_reduced_rhs = space_time_pod_basis.T.dot(rhs_no_bc[i+forwardstep+1])
        forwarded_reduced_rhs =  reduce_vector(rhs_no_bc[i+forwardstep+1],pod_basis)
        forwarded_reduced_rhs -= reduced_jump_matrix.dot(forwarded_reduced_solutions[-1])
        # forwarded_reduced_solutions.append(np.linalg.solve(reduced_system_matrix, forwarded_reduced_rhs))
        forwarded_reduced_solutions.append(scipy.linalg.lu_solve((LU_primal, piv_primal), forwarded_reduced_rhs))
        # forwarded_reduced_solutions.append(space_time_pod_basis.T.dot(primal_solutions[i+forwardstep+1]))
           
    # reversed forward dual solve
    forwarded_reduced_dual_solutions = []
    forwarded_reduced_dual_rhs = 2*reduced_mass_matrix_no_bc.dot(forwarded_reduced_solutions[-1])
    # forwarded_reduced_dual_solutions.append(np.linalg.solve(reduced_dual_matrix,forwarded_reduced_dual_rhs))
    forwarded_reduced_dual_solutions.append(scipy.linalg.lu_solve((LU_dual, piv_dual), forwarded_reduced_dual_rhs))
    # print(len(forwarded_reduced_solutions)-1)

    for forwardstep in range(2,len(forwarded_reduced_solutions)+1,1):
    # range(len(forwarded_reduced_solutions)-1,0,-1):
        forwarded_reduced_dual_rhs = 2*reduced_mass_matrix_no_bc.dot(forwarded_reduced_solutions[-forwardstep]) #len(forwarded_reduced_solutions)-forwardstep])
        forwarded_reduced_dual_rhs -= reduced_dual_jump_matrix_no_bc.T.dot(forwarded_reduced_dual_solutions[-1])
        # forwarded_reduced_dual_solutions.append(np.linalg.solve(reduced_dual_matrix,forwarded_reduced_dual_rhs))
        forwarded_reduced_dual_solutions.append(scipy.linalg.lu_solve((LU_dual, piv_dual), forwarded_reduced_dual_rhs))

    if len(forwarded_reduced_dual_solutions) == 1:
        forwarded_reduced_dual_solutions.append(forwarded_reduced_dual_solutions[-1])
        forwarded_reduced_dual_solutions[-2] = np.zeros_like(forwarded_reduced_dual_solutions[-1])
    # if i != n_slabs-1 and i != n_slabs-2 :
    #     reduced_dual_solutions = forwarded_reduced_dual_solutions[-3]
    # else:
    reduced_dual_solutions = forwarded_reduced_dual_solutions[-1]

    # projected_reduced_dual_solutions.append(space_time_pod_basis_dual.dot(reduced_dual_solutions))
    projected_reduced_dual_solutions.append(project_vector(reduced_dual_solutions,pod_basis_dual))
    extime_dual_solve += time.time() - start_time
              
    
    
    # # check dual FOM residual 
    if i>0:
        dual_residual.append(0)
    #     F_residual = 2*mass_matrix_no_bc.dot(projected_reduced_solutions[-1])
    #     F_residual -= jump_matrix_no_bc.T.dot(space_time_pod_basis_dual.dot(forwarded_reduced_dual_solutions[-2]))
    #     A_residual = dual_matrix_no_bc.dot(projected_reduced_dual_solutions[-1])
    #     dual_residual.append(np.linalg.norm(-A_residual + F_residual))

        
    # if dual_residual[-1] > tol_dual and i < n_slabs-1:
    #     last_projected_dual_solution = space_time_pod_basis_dual.dot(forwarded_reduced_dual_solutions[-2])
    #     pod_basis_dual, space_time_pod_basis_dual, reduced_dual_matrix, reduced_dual_jump_matrix_no_bc, projected_reduced_dual_solutions[-1], singular_values_dual, total_energy_dual = ROM_update_dual(
    #                  pod_basis_dual, 
    #                  space_time_pod_basis_dual, 
    #                  reduced_dual_matrix, 
    #                  reduced_dual_jump_matrix_no_bc, #reduced_dual_jump_matrix*0, 
    #                  last_projected_dual_solution,#forwarded_reduced_dual_solutions[-2], 
    #                  2*mass_matrix_no_bc.dot(projected_reduced_solutions[-1]),
    #                  jump_matrix_no_bc,
    #                  boundary_ids,
    #                  dual_matrix,
    #                  singular_values_dual,
    #                  total_energy_dual,
    #                  n_dofs,
    #                  time_dofs_per_time_interval,
    #                  dual_matrix_no_bc,
    #                  ENERGY_DUAL)    
    #     reduced_mass_matrix_no_bc = space_time_pod_basis_dual.T.dot(mass_matrix_no_bc.dot(space_time_pod_basis)).toarray()
    #     forwarded_reduced_dual_solutions[-2] = space_time_pod_basis_dual.T.dot(last_projected_dual_solution)
    #     # test residual again
    #     A_residual = dual_matrix_no_bc.dot(projected_reduced_dual_solutions[-1])
    #     dual_residual[-1] = np.linalg.norm(-A_residual + F_residual)
    # projected_reduced_dual_solutions.append(dual_solutions[i])

    # print(len(forwarded_reduced_solutions))
    # print(len(forwarded_reduced_dual_solutions)) 
    # print(" ")
    # reduced_solutions_old = reduced_solutions
    # dual_rhs = mass_matrix_no_bc.dot(projected_reduced_solutions[-1])
    # # dual_rhs -= reduced_dual_jump_matrix_no_bc.T.dot(last_dual_solution)
    # for row in boundary_ids:
    #     dual_rhs[row] = 0.
        
    # projected_reduced_dual_solutions.append(dual_solutions[i])
    # projected_reduced_dual_solutions.append(scipy.sparse.linalg.spsolve(dual_matrix, dual_rhs))

    
    # temporal localization of ROM error (using FOM dual solution)    
    start_time = time.time()
    tmp = -matrix_no_bc.dot(projected_reduced_solutions[i]) + rhs_no_bc[i]
    if i > 0:
        tmp -= jump_matrix_no_bc.dot(projected_reduced_solutions[i - 1])
    # temporal_interval_error.append(np.dot(dual_solutions[i], tmp))
    temporal_interval_error.append(np.dot(projected_reduced_dual_solutions[i], tmp))
    temporal_interval_error_relative.append(np.abs(temporal_interval_error[-1])/np.abs(np.dot(projected_reduced_solutions[-1],  mass_matrix_no_bc.dot(projected_reduced_solutions[-1]))+temporal_interval_error[-1]))
    temporal_interval_error_incidactor.append(0)
    # or  np.abs(temporal_interval_error[-1]/temporal_interval_error[i-1]):
    extime_error += time.time() - start_time
        
    start_time = time.time()
    # if np.abs(temporal_interval_error[-1]) > tol:
    if temporal_interval_error_relative[-1] > tol_rel:
    # np.abs(temporal_interval_error[-1])/np.abs(np.dot(projected_reduced_solutions[-1],  mass_matrix_no_bc.dot(projected_reduced_solutions[-1]))+temporal_interval_error[-1]) > tol_rel:
        temporal_interval_error_incidactor[-1] = 1
        # print(np.linalg.norm(projected_reduced_solutions[-1]))
        # pod_basis,  space_time_pod_basis, reduced_system_matrix, reduced_jump_matrix, projected_reduced_solutions[-1], singular_values, total_energy = ROM_update(
        pod_basis, reduced_system_matrix, reduced_jump_matrix, projected_reduced_solutions[-1], singular_values, total_energy = ROM_update(
                     pod_basis, 
                     # space_time_pod_basis, 
                     reduced_system_matrix, 
                     reduced_jump_matrix, 
                     projected_reduced_solutions[i - 1], 
                     rhs_no_bc[i].copy(),
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
        forwarded_reduced_solutions.append(reduce_vector(projected_reduced_solutions[-1],pod_basis))
        # forwarded_reduced_solutions.append(space_time_pod_basis.T.dot(primal_solutions[i]))

        for forwardstep in range(forwardsteps):
            if i+forwardstep+1 >= n_slabs:
                break
            forwarded_reduced_rhs =  reduce_vector(rhs_no_bc[i+forwardstep+1],pod_basis)
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
        
        
        
        # pod_basis_dual, space_time_pod_basis_dual, reduced_dual_matrix, reduced_dual_jump_matrix_no_bc, projected_reduced_dual_solutions[-1], singular_values_dual, total_energy_dual = ROM_update_dual(
        pod_basis_dual, reduced_dual_matrix, reduced_dual_jump_matrix_no_bc, projected_reduced_dual_solutions[-1], singular_values_dual, total_energy_dual = ROM_update_dual(
                     pod_basis_dual, 
                     # space_time_pod_basis_dual, 
                     reduced_dual_matrix, 
                     reduced_dual_jump_matrix_no_bc, #reduced_dual_jump_matrix*0, 
                     # space_time_pod_basis_dual.dot(forwarded_reduced_dual_solutions[-2]), 
                     project_vector(forwarded_reduced_dual_solutions[-2],pod_basis_dual), 
                     # projected_reduced_dual_solutions[-2],
                     2*mass_matrix_no_bc.dot( projected_reduced_solutions[-1]),
                     jump_matrix_no_bc,
                     boundary_ids,
                     dual_matrix,
                     singular_values_dual,
                     total_energy_dual,
                     n_dofs,
                     time_dofs_per_time_interval,
                     dual_matrix_no_bc,
                     ENERGY_DUAL)    
        # reduced_mass_matrix_no_bc = space_time_pod_basis_dual.T.dot(mass_matrix_no_bc.dot(space_time_pod_basis)).toarray()
        reduced_mass_matrix_no_bc = reduce_matrix(mass_matrix_no_bc, pod_basis_dual, pod_basis)

        
        # Lu decompostion of reduced matrices
        LU_dual, piv_dual     = scipy.linalg.lu_factor(reduced_dual_matrix)
        
        
        # reduced_dual_jump_matrix_no_bc = space_time_pod_basis_dual.T.dot(jump_matrix_no_bc.dot(space_time_pod_basis_dual)).toarray()
        
        # print(np.linalg.norm(projected_reduced_solutions[-1])) 
        # reduced_solutions = space_time_pod_basis.T.dot(projected_reduced_solutions[-1])
        reduced_solutions = reduce_vector(projected_reduced_solutions[-1], pod_basis)
        
        # tmp = -matrix_no_bc.dot(projected_reduced_solutions[i]) + rhs_no_bc[i]
        # if i > 0:
        #     tmp -= jump_matrix_no_bc.dot(projected_reduced_solutions[i - 1])
        # temporal_interval_error[-1] = np.dot(dual_solutions[i], tmp)
        
        # temporal_interval_error[-1] = np.dot(projected_reduced_dual_solutions[-1], tmp)
        
        # if np.abs(temporal_interval_error[-1])/np.abs(np.dot(projected_reduced_solutions[-1],  mass_matrix_no_bc.dot(projected_reduced_solutions[-1]))+temporal_interval_error[-1]) > tol_rel:
        # if np.abs(np.dot(projected_reduced_dual_solutions[-1], tmp))/np.abs(np.dot(projected_reduced_solutions[-1],  mass_matrix_no_bc.dot(projected_reduced_solutions[-1]))+temporal_interval_error[-1]) > tol_rel:
        #     print('Error correction failed')
    extime_update += time.time() - start_time

    reduced_solutions_old = reduced_solutions
    
    
end_execution = time.time()
execution_time_ROM = end_execution - start_execution
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
print("Error est time:      " + str(extime_error))
print("Update time:         " + str(extime_update))
print("Overall time:        " + str(extime_solve+extime_error+extime_update+extime_dual_solve))
print(" ")
projected_reduced_solution = np.hstack(projected_reduced_solutions)

original_stdout = sys.stdout  # Save a reference to the original standard output
# with open('output/speedup_' + CASE + '_cycle_' + str(cycle) + '.txt', 'a') as f:
#     sys.stdout = f # Change the standard output to the file we created.
#     print(str(execution_time_FOM) + ', ' + str(execution_time_ROM) + ', ' + str(execution_time_FOM/execution_time_ROM))
#     sys.stdout = original_stdout # Reset the standard output to its original value


J_r = 0.
J_r_t = np.empty([n_slabs, 1])
J_r_t_before_enrichement = np.empty([n_slabs, 1])
for i in range(n_slabs):
    J_r_t[i] = np.dot(projected_reduced_solutions[i], mass_matrix_no_bc.dot(projected_reduced_solutions[i]))
    J_r_t_before_enrichement[i] = np.dot(projected_reduced_solutions_before_enrichment[i], mass_matrix_no_bc.dot(projected_reduced_solutions_before_enrichment[i]))
    
J["u_r"] = np.sum(J_r_t) 

print("J(u_h) =", J["u_h"])
# TODO: in the future compare J(u_r) for different values of rprojected_reduced_dual_solutions
print("J(u_r) =", J["u_r"])
print(" ")

# %% error estimation
true_error = J['u_h'] - J['u_r']

error_tol = np.abs((J_h_t - J_r_t)/J_h_t) > tol_rel

#print(np.sum(error_tol.astype(int)))

J_r_t = J_r_t_before_enrichement

true_tol = np.abs((J_h_t - J_r_t)/J_h_t) > tol_rel
esti_tol = np.abs(1.*np.array(temporal_interval_error).reshape(-1,1)/(J_r_t_before_enrichement+np.array(temporal_interval_error).reshape(-1,1))) > tol_rel

#esti_tol = np.abs(np.array(temporal_interval_error).reshape(-1,1)/J_h_t) > tol_rel

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
for i, error in enumerate(temporal_interval_error_relative):
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
plt.plot(xx, yy, label="$u_N$")
plt.plot(xx_FOM, yy_FOM, color='r', label="$u_h$")
plt.grid()
plt.legend()
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlabel('$t \; [$s$]$')
plt.ylabel("$\eta_{\rel}\\raisebox{-.5ex}{$|$}_{Q_l}$")
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

# plot temporal evolution of dual resisudal
plt.rcParams["figure.figsize"] = (10,2)
plt.plot(np.arange(0, n_slabs*time_step_size, time_step_size),
         dual_residual, c='#1f77b4', label="dual residual")
plt.grid()
plt.legend()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.xlabel('$t \; [$s$]$')
plt.ylabel("$dual residual$")
plt.xlim([0, n_slabs*time_step_size])
#plt.title("temporal evaluation of cost funtional")
# plt.savefig(SAVE_PATH + "error_estimate_over_time.eps", format='eps')
# plt.savefig(SAVE_PATH + "error_estimate_over_time.png", format='png')
plt.show()

# %%
# for dual_solution in dual_solutions:
if PLOTTING:
    primal_max = 0.0
    primal_min = 0.0
    for primal_solution in primal_solutions:
        primal_max = np.max([primal_max,np.max(primal_solution)])
        primal_min = np.min([primal_min,np.min(primal_solution)])
        
    dual_max = 0.0
    dual_min = 0.0
    for dual_solution in dual_solutions:
        dual_max = np.max([dual_max,np.max(dual_solution)])
        dual_min = np.min([dual_min,np.min(dual_solution)])    
        
    print(f"primal = {primal_min} - {primal_max}")
    print(f"dual   = {dual_min} - {dual_max}")
    i = 0
    for (primal_solution,projected_reduced_solution, dual_solution, projected_reduced_dual_solution) in zip(primal_solutions,projected_reduced_solutions,dual_solutions, projected_reduced_dual_solutions):        
        grid_x, grid_y = np.mgrid[0:1:50j, 0:1:50j]
        dual_grid = scipy.interpolate.griddata(
            coordinates_x.T, dual_solution[0:n_dofs["space"]], (grid_x, grid_y), method=INTERPOLATION_TYPE)
        # fig, _ = plt.subplots(figsize=(15,15))
        
        fig, ((ax3, ax4),(ax1,ax2)) = plt.subplots(2, 2, figsize=(30,30))
        
        # ax1.set_title(f"Dual solution (ref={cycle.split('=')[1]})")
        im1 = ax1.imshow(dual_grid.T, extent=(0, 1, 0, 1), origin='lower',vmin=dual_min,vmax=dual_max)
        # ax1.set_clim([-0.00025, 0.00025])
        # ax1.set_xlabel("$y$")
        # ax1.set_ylabel("$x$")
        # ax1.colorbar()
        # plt.colorbar(im1,ax=ax1)
        
        dual_grid_reduced = scipy.interpolate.griddata(
            coordinates_x.T, projected_reduced_dual_solution[0:n_dofs["space"]], (grid_x, grid_y), method=INTERPOLATION_TYPE)
        # ax2.set_title(f"Dual solution (ref={cycle.split('=')[1]})")
        im2 = ax2.imshow(dual_grid_reduced.T, extent=(0, 1, 0, 1), origin='lower',vmin=dual_min,vmax=dual_max)
        # ax2.set_clim(-0.00025, 0.00025)
        # ax2.set_xlabel("$y$")
        # ax2.set_ylabel("$x$")
        # ax2.set_colorbar()
        # plt.colorbar(im2,ax=ax2)
        # plt.show()
        
        primal_grid = scipy.interpolate.griddata(
            coordinates_x.T, primal_solution[0:n_dofs["space"]], (grid_x, grid_y), method=INTERPOLATION_TYPE)
        # fig, _ = plt.subplots(figsize=(15,15))
        # ax3.set_title(f"Dual solution (ref={cycle.split('=')[1]})")
        im3 = ax3.imshow(primal_grid.T, extent=(0, 1, 0, 1), origin='lower' ,vmin=primal_min,vmax=primal_max)
        # ax1.set_clim([-0.00025, 0.00025])
        # ax3.set_xlabel("$y$")
        # ax3.set_ylabel("$x$")
        # ax1.colorbar()
        # plt.colorbar(im1,ax=ax1)
        
        reduced_primal_grid = scipy.interpolate.griddata(
            coordinates_x.T, projected_reduced_solution[0:n_dofs["space"]], (grid_x, grid_y), method=INTERPOLATION_TYPE)
        # ax4.set_title(f"Dual solution (ref={cycle.split('=')[1]})")
        im4 = ax4.imshow(reduced_primal_grid.T, extent=(0, 1, 0, 1), origin='lower',vmin=primal_min,vmax=primal_max)
        # ax2.set_clim(-0.00025, 0.00025)
        # ax4.set_xlabel("$y$")
        # ax4.set_ylabel("$x$")
        plt.show()
        fig.savefig(SAVE_PATH + f"movie/movie{i}.png", format='png')
        i += 1
                            # f"/primal_solution{i:05}.vtk    
    
    
    # def dual_gif(primal_solution, projected_reduced_solution, dual_solution, projected_reduced_dual_solution):
    #     grid_x, grid_y = np.mgrid[0:1:50j, 0:1:50j]
    #     dual_grid = scipy.interpolate.griddata(
    #         coordinates_x.T, dual_solution[0:n_dofs["space"]], (grid_x, grid_y), method=INTERPOLATION_TYPE)
    #     # fig, _ = plt.subplots(figsize=(15,15))
        
    #     fig, ((ax3, ax4),(ax1,ax2)) = plt.subplots(2, 2, figsize=(30,30))
        
    #     ax1.set_title(f"Dual solution (ref={cycle.split('=')[1]})")
    #     im1 = ax1.imshow(dual_grid.T, extent=(0, 1, 0, 1), origin='lower',vmin=dual_min,vmax=dual_max)
    #     # ax1.set_clim([-0.00025, 0.00025])
    #     ax1.set_xlabel("$y$")
    #     ax1.set_ylabel("$x$")
    #     # ax1.colorbar()
    #     # plt.colorbar(im1,ax=ax1)
        
    #     dual_grid_reduced = scipy.interpolate.griddata(
    #         coordinates_x.T, projected_reduced_dual_solution[0:n_dofs["space"]], (grid_x, grid_y), method=INTERPOLATION_TYPE)
    #     ax2.set_title(f"Dual solution (ref={cycle.split('=')[1]})")
    #     im2 = ax2.imshow(dual_grid_reduced.T, extent=(0, 1, 0, 1), origin='lower',vmin=dual_min,vmax=dual_max)
    #     # ax2.set_clim(-0.00025, 0.00025)
    #     ax2.set_xlabel("$y$")
    #     ax2.set_ylabel("$x$")
    #     # ax2.set_colorbar()
    #     # plt.colorbar(im2,ax=ax2)
    #     # plt.show()
        
    #     primal_grid = scipy.interpolate.griddata(
    #         coordinates_x.T, primal_solution[0:n_dofs["space"]], (grid_x, grid_y), method=INTERPOLATION_TYPE)
    #     # fig, _ = plt.subplots(figsize=(15,15))
    #     ax3.set_title(f"Dual solution (ref={cycle.split('=')[1]})")
    #     im3 = ax3.imshow(primal_grid.T, extent=(0, 1, 0, 1), origin='lower',vmin=primal_min,vmax=primal_max)
    #     # ax1.set_clim([-0.00025, 0.00025])
    #     ax3.set_xlabel("$y$")
    #     ax3.set_ylabel("$x$")
    #     # ax1.colorbar()
    #     # plt.colorbar(im1,ax=ax1)
        
    #     reduced_primal_grid = scipy.interpolate.griddata(
    #         coordinates_x.T, projected_reduced_solution[0:n_dofs["space"]], (grid_x, grid_y), method=INTERPOLATION_TYPE)
    #     ax4.set_title(f"Dual solution (ref={cycle.split('=')[1]})")
    #     im4 = ax4.imshow(reduced_primal_grid.T, extent=(0, 1, 0, 1), origin='lower',vmin=primal_min,vmax=primal_max)
    #     # ax2.set_clim(-0.00025, 0.00025)
    #     ax4.set_xlabel("$y$")
    #     ax4.set_ylabel("$x$")
    #     # ax2.set_colorbar()
    #     # plt.colorbar(im2,ax=ax2)
    #     # plt.show()
        
    #     fig.canvas.draw()       # draw the canvas, cache the renderer
    #     image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    #     image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    #     return image

    # imageio.mimsave('./solution.mp4', [dual_gif(primal_solution, projected_reduced_solution,dual_solution, projected_reduced_dual_solution) for (primal_solution,projected_reduced_solution, dual_solution, projected_reduced_dual_solution) in zip(primal_solutions,projected_reduced_solutions,dual_solutions, projected_reduced_dual_solutions)], fps=10)
    