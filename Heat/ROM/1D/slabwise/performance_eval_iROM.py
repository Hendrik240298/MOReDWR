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
import random
from iPOD_1D import iPOD, ROM_update, ROM_update_dual, reduce_matrix, reduce_vector, project_vector
# import imageio


PLOTTING = False
INTERPOLATION_TYPE = "cubic"  # "linear", "cubic"
CASE = ""  # "two" or "moving"
MOTHER_PATH = "/home/ifam/fischer/Code/MORe_DWR/Heat/"
MOTHER_PATH = "/home/hendrik/Code/MORe_DWR/Heat/"
OUTPUT_PATH = MOTHER_PATH + "Data/1D/moving_source/slabwise/FOM/"
cycle = "cycle=4-4"
SAVE_PATH = MOTHER_PATH + "Data/1D/moving_source/slabwise/ROM/" + cycle + "/"
# SAVE_PATH = cycle + "/output_ROM/"

# "../../FOM/slabwise/output_" + CASE + "/dim=1/"

identifier = random.randint(0, 99999)
identifier = format(identifier, '05d')
print(identifier)
# "../../FOM/slabwise/output_" + CASE + "/dim=1/"projected_reduced_solutions

# redirect terminjal output to file
# orig_stdout = sys.stdout
# f = open("out.txt", "w")
# sys.stdout = f

ENERGY_PRIMAL = 0.99999999
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

[data, row, column] = np.loadtxt(
    OUTPUT_PATH + cycle + "/mass_matrix_no_bc.txt")
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
print(f"coordinates_x.shape: {coordinates_x.shape}")
n_dofs = {"space": coordinates_x.shape[0], "time": coordinates_t.shape[0]}

# %% primal FOM solve
test_numbers = 20
execution_time_FOM_mean = 0
for test_number in range(test_numbers):
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
    execution_time_FOM_mean += execution_time_FOM/test_numbers
    print(f"execution_time_FOM: {execution_time_FOM}")
    # open file to write and append with execution_time_FOM
    with open(SAVE_PATH + "execution_time_FOM.txt", "a") as f:
        f.write(f"{execution_time_FOM}\n")

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
    J_h_t[i] = np.dot(primal_solutions[i],
                      mass_matrix_no_bc.dot(primal_solutions[i]))
J["u_h"] = np.sum(J_h_t)

# %%
time_dofs_per_time_interval = int(n_dofs["time"] / n_slabs)
# dofs_per_time_interval = time_dofs_per_time_interval * n_dofs["space"]


# %% init setting
tol = 1e-10
tol_rel = 10e-2
tol_dual = 5e-1
print("tol_rel       = " + str(tol_rel))
execution_time_ROM_mean = 0
for test_number in range(test_numbers):
    # %% initilaize ROM framework
    total_energy = 0
    pod_basis = np.empty([0, 0])
    bunch = np.empty([0, 0])
    singular_values = np.empty([0, 0])

    # only use first solution of slab since we assume that solutions are quite similar
    for slab_step in range(int(primal_solutions[0].shape[0] / n_dofs["space"])):
        # print(slab_step)
        pod_basis, bunch, singular_values, total_energy = iPOD(pod_basis,
                                                               bunch,
                                                               singular_values,
                                                               primal_solutions[0][range(
                                                                   slab_step * n_dofs["space"], (slab_step + 1) * n_dofs["space"])],
                                                               total_energy,
                                                               ENERGY_PRIMAL)

    # change from the FOM to the POD basis
    space_time_pod_basis = scipy.sparse.block_diag(
        [pod_basis] * time_dofs_per_time_interval)

    # print(pod_basis.shape)

    reduced_system_matrix = reduce_matrix(matrix_no_bc, pod_basis, pod_basis)
    reduced_jump_matrix = reduce_matrix(
        jump_matrix_no_bc, pod_basis, pod_basis)

    # %% initilaize dual ROM framework
    total_energy_dual = 0
    pod_basis_dual = np.empty([0, 0])
    bunch_dual = np.empty([0, 0])
    singular_values_dual = np.empty([0, 0])

    # print(space_time_pod_basis.shape)

    # onyl use first solution of slab since we assume that solutions are quite similar
    for slab_step in range(int(dual_solutions[0].shape[0] / n_dofs["space"])):
        pod_basis_dual, bunch_dual, singular_values_dual, total_energy_dual = iPOD(pod_basis_dual,
                                                                                   bunch_dual,
                                                                                   singular_values_dual,
                                                                                   dual_solutions[0][range(
                                                                                       slab_step * n_dofs["space"], (slab_step + 1) * n_dofs["space"])],
                                                                                   total_energy_dual,
                                                                                   ENERGY_DUAL)

    reduced_dual_matrix = reduce_matrix(
        dual_matrix_no_bc, pod_basis_dual, pod_basis_dual)
    reduced_dual_jump_matrix_no_bc = reduce_matrix(
        jump_matrix_no_bc, pod_basis_dual, pod_basis_dual)

    reduced_mass_matrix_no_bc = reduce_matrix(
        mass_matrix_no_bc, pod_basis_dual, pod_basis)

    reduced_matrix_no_bc_estimator = reduce_matrix(
        matrix_no_bc, pod_basis_dual, pod_basis)
    reduced_jump_matrix_no_bc_estimator = reduce_matrix(
        jump_matrix_no_bc, pod_basis_dual, pod_basis)
    reduced_mass_matrix_no_bc_cst_fct = reduce_matrix(
        mass_matrix_no_bc, pod_basis, pod_basis)

    # %% primal ROM solve

    LU_primal, piv_primal = scipy.linalg.lu_factor(reduced_system_matrix)
    LU_dual, piv_dual = scipy.linalg.lu_factor(reduced_dual_matrix)

    reduced_solutions = []
    # reduced_solutions_old = space_time_pod_basis.T.dot(np.zeros_like(primal_solutions[0]))
    reduced_solutions_old = reduce_vector(
        np.zeros_like(primal_solutions[0]), pod_basis)

    reduced_dual_solutions = []
    # reduced_dual_solutions_old = space_time_pod_basis_dual.T.dot(np.zeros_like(dual_solutions[0]))
    reduced_dual_solutions_old = reduce_vector(
        np.zeros_like(dual_solutions[0]), pod_basis)

    projected_reduced_solutions = []
    projected_reduced_dual_solutions = []

    dual_residual = []
    dual_residual.append(0)

    temporal_interval_error = []
    temporal_interval_error_relative = []
    temporal_interval_error_incidactor = []

    index_fom_solves = []
    record_basis_size = []
    record_dual_basis_size = []

    # print("tol =     " + str(tol))

    extime_solve = 0.0
    extime_dual_solve = 0.0
    extime_error = 0.0
    extime_update = 0.0
    # count number of FOM solves during iROM loop: PRIMAL AND DUAL COMBINED
    number_FOM_solves = 0

    # 64*2*2# 32 # int(2*slab_properties["n_total"]/len_block_evaluation)
    nb_buckets = 32  # 1
    len_block_evaluation = int(n_slabs/nb_buckets)

    forwardsteps = len_block_evaluation

    # print("tol           = " + str(tol))
    # print(f"forward steps = {forwardsteps}")
    # print(" ")

    start_execution = time.time()

    projected_reduced_solutions_buckets_combined = []
    reduced_solutions_buckets_combined = []
    temporal_interval_error_incidactor_combinded = []
    temporal_interval_error_combinded = []
    temporal_interval_error_relative_combinded = []
    index_primal = 0
    last_bucket_end_solution = np.zeros(matrix_no_bc.shape[0])
    for it_bucket in range(nb_buckets):
        # print("bucket " + str(it_bucket+1) + " of " + str(nb_buckets) + " of length: " + str(len_block_evaluation))
        bucket_shift = it_bucket*len_block_evaluation
        temporal_interval_error_incidactor = np.zeros(len_block_evaluation)

        while True:
            # primal ROM solve
            extime_solve_start = time.time()
            primal_reduced_solutions = [reduce_vector(
                last_bucket_end_solution, pod_basis)]
            for i in range(len_block_evaluation):
                reduced_rhs = reduce_vector(
                    rhs_no_bc[i+bucket_shift], pod_basis)
                reduced_rhs -= reduced_jump_matrix.dot(
                    primal_reduced_solutions[-1])
                primal_reduced_solutions.append(
                    scipy.linalg.lu_solve((LU_primal, piv_primal), reduced_rhs))
            primal_reduced_solutions = primal_reduced_solutions[1:]
            extime_solve += time.time() - extime_solve_start

            # reversed forward dual solve
            extime_dual_solve_start = time.time()
            reduced_dual_solutions = [
                np.zeros(reduced_dual_jump_matrix_no_bc.shape[1])]
            for i in range(len_block_evaluation):
                # len(forwarded_reduced_solutions)-forwardstep])
                reduced_dual_rhs = 2 * \
                    reduced_mass_matrix_no_bc.dot(primal_reduced_solutions[-i])
                reduced_dual_rhs -= reduced_dual_jump_matrix_no_bc.T.dot(
                    reduced_dual_solutions[-1])
                reduced_dual_solutions.append(scipy.linalg.lu_solve(
                    (LU_dual, piv_dual), reduced_dual_rhs))
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
                tmp = -reduced_matrix_no_bc_estimator.dot(primal_reduced_solutions[i]) + reduce_vector(
                    rhs_no_bc[i+bucket_shift], pod_basis_dual)
                if i > 0:
                    tmp -= reduced_jump_matrix_no_bc_estimator.dot(
                        primal_reduced_solutions[i - 1])
                else:
                    # reduce_vector(jump_matrix_no_bc.dot(last_bucket_end_solution),pod_basis_dual) #
                    tmp -= reduced_jump_matrix_no_bc_estimator.dot(
                        reduce_vector(last_bucket_end_solution, pod_basis))
                temporal_interval_error.append(
                    np.dot(reduced_dual_solutions[i], tmp))
                goal_functional = np.dot(primal_reduced_solutions[i],  reduced_mass_matrix_no_bc_cst_fct.dot(
                    primal_reduced_solutions[i]))
                temporal_interval_error_relative.append(np.abs(
                    temporal_interval_error[-1])/np.abs(goal_functional+temporal_interval_error[-1]))
            extime_error += time.time() - extime_error_start

            # temporal_interval_error_relative = temporal_interval_error
            estimated_error = np.max(np.abs(temporal_interval_error_relative))

            # print(estimated_error)
            if estimated_error < tol_rel:
                # if estimated_error < tol:
                break
            else:
                extime_update_start = time.time()
                index_primal = np.argmax(
                    np.abs(temporal_interval_error_relative))
                index_dual = index_primal
                # print(str(index_primal) + ":   " + str(np.abs(temporal_interval_error_relative[index_primal])))
                # print(" ")
                index_fom_solves.append(
                    [index_primal+bucket_shift, estimated_error])
                temporal_interval_error_incidactor[index_primal] = 1
                # print(f"i:            {i}")
                # print(f"index_primal: {index_primal}")
                if index_primal > 0:
                    # projected_reduced_solutions[index_primal - 1]
                    old_projected_solution = project_vector(
                        primal_reduced_solutions[index_primal-1], pod_basis)
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
                LU_primal, piv_primal = scipy.linalg.lu_factor(
                    reduced_system_matrix)
                number_FOM_solves += 1

                reduced_mass_matrix_no_bc = reduce_matrix(
                    mass_matrix_no_bc, pod_basis_dual, pod_basis)

                forwarded_reduced_solutions = []
                forwarded_reduced_solutions.append(
                    reduce_vector(new_projection_solution, pod_basis))

                for forwardstep in range(forwardsteps):
                    if index_primal+forwardstep+1 + bucket_shift >= n_slabs:
                        break
                    forwarded_reduced_rhs = reduce_vector(
                        rhs_no_bc[index_primal+forwardstep+1+bucket_shift], pod_basis)
                    forwarded_reduced_rhs -= reduced_jump_matrix.dot(
                        forwarded_reduced_solutions[-1])
                    forwarded_reduced_solutions.append(scipy.linalg.lu_solve(
                        (LU_primal, piv_primal), forwarded_reduced_rhs))

                # reversed forward dual solve
                forwarded_reduced_dual_solutions = []
                forwarded_reduced_dual_rhs = 2 * \
                    reduced_mass_matrix_no_bc.dot(
                        forwarded_reduced_solutions[-1])
                forwarded_reduced_dual_solutions.append(scipy.linalg.lu_solve(
                    (LU_dual, piv_dual), forwarded_reduced_dual_rhs))

                for forwardstep in range(2, len(forwarded_reduced_solutions)+1, 1):
                    # len(forwarded_reduced_solutions)-forwardstep])
                    forwarded_reduced_dual_rhs = 2 * \
                        reduced_mass_matrix_no_bc.dot(
                            forwarded_reduced_solutions[-forwardstep])
                    forwarded_reduced_dual_rhs -= reduced_dual_jump_matrix_no_bc.T.dot(
                        forwarded_reduced_dual_solutions[-1])
                    forwarded_reduced_dual_solutions.append(scipy.linalg.lu_solve(
                        (LU_dual, piv_dual), forwarded_reduced_dual_rhs))

                if len(forwarded_reduced_dual_solutions) == 1:
                    forwarded_reduced_dual_solutions.append(
                        forwarded_reduced_dual_solutions[-1])
                    forwarded_reduced_dual_solutions[-2] = np.zeros_like(
                        forwarded_reduced_dual_solutions[-1])

                reduced_dual_solutions = forwarded_reduced_dual_solutions[-1]

                pod_basis_dual, reduced_dual_matrix, reduced_dual_jump_matrix_no_bc, _, singular_values_dual, total_energy_dual = ROM_update_dual(
                    pod_basis_dual,
                    # space_time_pod_basis_dual,
                    reduced_dual_matrix,
                    reduced_dual_jump_matrix_no_bc,  # reduced_dual_jump_matrix*0,
                    # space_time_pod_basis_dual.dot(forwarded_reduced_dual_solutions[-2]),
                    project_vector(
                        forwarded_reduced_dual_solutions[-2], pod_basis_dual),
                    # projected_reduced_dual_solutions[-2],
                    2*mass_matrix_no_bc.dot(new_projection_solution),
                    jump_matrix_no_bc,
                    boundary_ids,
                    dual_matrix,
                    singular_values_dual,
                    total_energy_dual,
                    n_dofs,
                    time_dofs_per_time_interval,
                    dual_matrix_no_bc,
                    ENERGY_DUAL)
                reduced_mass_matrix_no_bc = reduce_matrix(
                    mass_matrix_no_bc, pod_basis_dual, pod_basis)
                number_FOM_solves += 1
                reduced_matrix_no_bc_estimator = reduce_matrix(
                    matrix_no_bc, pod_basis_dual, pod_basis)
                reduced_jump_matrix_no_bc_estimator = reduce_matrix(
                    jump_matrix_no_bc, pod_basis_dual, pod_basis)
                reduced_mass_matrix_no_bc_cst_fct = reduce_matrix(
                    mass_matrix_no_bc, pod_basis, pod_basis)
                # Lu decompostion of reduced matrices
                LU_dual, piv_dual = scipy.linalg.lu_factor(reduced_dual_matrix)
                extime_update += time.time() - extime_update_start

                last_bucket_end_solution = project_vector(
                    reduce_vector(last_bucket_end_solution, pod_basis), pod_basis)

        index_primal = len_block_evaluation-1

        last_bucket_end_solution = project_vector(
            primal_reduced_solutions[-1], pod_basis)  # projected_reduced_solutions[-1]

        record_basis_size.append((pod_basis.shape[1]))
        record_dual_basis_size.append((pod_basis_dual.shape[1]))
        for i in range(len_block_evaluation):
            #     reduced_solutions_buckets_combined.append(primal_reduced_solutions[i])
            temporal_interval_error_incidactor_combinded.append(
                temporal_interval_error_incidactor[i])
        #     temporal_interval_error_combinded.append(temporal_interval_error[i])
        #     temporal_interval_error_relative_combinded.append(temporal_interval_error_relative[i])

    # %% ---------------------- VERIFICATION ------------------------------------------------------------

    start_time = time.time()
    primal_reduced_solutions = [reduce_vector(
        np.zeros(matrix_no_bc.shape[0]), pod_basis)]
    for i in range(n_slabs):
        reduced_rhs = reduce_vector(rhs_no_bc[i], pod_basis)
        reduced_rhs -= reduced_jump_matrix.dot(primal_reduced_solutions[-1])
        primal_reduced_solutions.append(
            scipy.linalg.lu_solve((LU_primal, piv_primal), reduced_rhs))
    primal_reduced_solutions = primal_reduced_solutions[1:]
    extime_solve += time.time() - extime_solve_start

    # reversed forward dual solve
    extime_dual_solve_start = time.time()
    reduced_dual_solutions = [
        np.zeros(reduced_dual_jump_matrix_no_bc.shape[1])]
    for i in range(n_slabs):
        # len(forwarded_reduced_solutions)-forwardstep])
        reduced_dual_rhs = 2 * \
            reduced_mass_matrix_no_bc.dot(primal_reduced_solutions[-i])
        reduced_dual_rhs -= reduced_dual_jump_matrix_no_bc.T.dot(
            reduced_dual_solutions[-1])
        reduced_dual_solutions.append(scipy.linalg.lu_solve(
            (LU_dual, piv_dual), reduced_dual_rhs))
    reduced_dual_solutions = reduced_dual_solutions[1:]
    reduced_dual_solutions = reduced_dual_solutions[::-1]
    extime_dual_solve += time.time() - extime_dual_solve_start

    extime_error_start = time.time()
    temporal_interval_error = []
    temporal_interval_error_relative = []
    J_r_t = np.empty([n_slabs, 1])

    for i in range(n_slabs):
        tmp = -reduced_matrix_no_bc_estimator.dot(
            primal_reduced_solutions[i]) + reduce_vector(rhs_no_bc[i], pod_basis_dual)
        if i > 0:
            tmp -= reduced_jump_matrix_no_bc_estimator.dot(
                primal_reduced_solutions[i - 1])
        else:
            tmp -= reduced_jump_matrix_no_bc_estimator.dot(
                reduce_vector(np.zeros(matrix_no_bc.shape[0]), pod_basis))
        temporal_interval_error.append(np.dot(reduced_dual_solutions[i], tmp))
        J_r_t[i] = np.dot(primal_reduced_solutions[i],
                          reduced_mass_matrix_no_bc_cst_fct.dot(primal_reduced_solutions[i]))
        temporal_interval_error_relative.append(np.abs(
            temporal_interval_error[-1])/np.abs(J_r_t[i]+temporal_interval_error[-1]))
    extime_error += time.time() - extime_error_start
    time_verification = time.time() - start_time

    end_execution = time.time()
    execution_time_ROM = end_execution - start_execution
    execution_time_ROM_mean += execution_time_ROM/test_numbers
    print(f"execution_time_ROM: {execution_time_ROM}")
    # open file to write and append with execution_time_FOM
    with open(SAVE_PATH + f"execution_time_ROM_{tol_rel}.txt", "a") as f:
        f.write(f"{execution_time_ROM}\n")

estimated_error = np.max(np.abs(temporal_interval_error_relative))

# print(f"Largest Error in Verification: {estimated_error}")


# %% Simulation Data
temporal_interval_error_incidactor = temporal_interval_error_incidactor_combinded

print("FOM time - mean:                      " + str(execution_time_FOM_mean))
print(f"ROM time ({tol_rel}) - mean:         " + str(execution_time_ROM_mean))
print(f"speedup: ({tol_rel}) act/max: " + str(execution_time_FOM_mean/execution_time_ROM_mean) + " / " +
      str(len(temporal_interval_error_incidactor)/(2*np.sum(temporal_interval_error_incidactor))))
