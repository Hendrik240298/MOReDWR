from tabulate import tabulate
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
import random

from auxiliaries import save_vtk, read_in_LES, apply_boundary_conditions, read_in_discretization
from auxiliaries import solve_primal_FOM_step, solve_dual_FOM_step, solve_primal_ROM_step, reorder_matrix
from auxiliaries import reorder_vector, error_estimator, save_solution_txt, load_solution_txt
from auxiliaries import evaluate_cost_functional, find_solution_indicies_for_slab, plot_matrix


MOTHER_PATH = "/home/ifam/fischer/Code/MORe_DWR_Revision2/MORe_DWR/Elastodynamics/"
OUTPUT_PATH = MOTHER_PATH + "/Data/3D/Rod/"
OUTPUT_PATH_DUAL = MOTHER_PATH + "Dual_Elastodynamics/Data/3D/Rod/"
LOG_PATH = MOTHER_PATH + "ROM/"

# Check if file exists then delete it
if os.path.exists(LOG_PATH + "discretization.log"):
    os.remove(LOG_PATH + "discretization.log")
else:
    print("The file does not exist")

space_cycles = 4
time_cycles = 5

space_label = []
time_label = []
# for time_cycle in range(time_cycles):
#     time_label.append(20 * 2**time_cycle)

J_h = np.zeros([space_cycles, time_cycles])

for time_cycle in range(time_cycles):
    for space_cycle in range(space_cycles):
        cycle = f"cycle={space_cycle}-{time_cycle}"
        print(cycle)
        SAVE_PATH = MOTHER_PATH + "Data/ROM/" + cycle + "/"

        LOAD_SOLUTION = False

        # if SAVE_PATH directory not exists create it
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)

        # %% read in properties connected to discretization
        n_dofs, slab_properties, index2measuredisp, dof_matrix, grid = read_in_discretization(
            OUTPUT_PATH + cycle)

        # since we use whole block system
        slab_properties["n_time_unknowns"] += 1
        # print("n_time_unknowns", slab_properties["n_time_unknowns"])

# %% Reading in matrices and rhs without bc
        matrix_no_bc, rhs_no_bc = read_in_LES(
            OUTPUT_PATH + cycle, "/matrix_no_bc.txt", "primal_rhs_no_bc")
        mass_matrix_no_bc, _ = read_in_LES(
            OUTPUT_PATH + cycle, "/mass_matrix_no_bc.txt", "primal_rhs_no_bc")

        _, dual_rhs_no_bc = read_in_LES(
            OUTPUT_PATH + cycle, "/matrix_no_bc.txt", "dual_rhs_no_bc")

        matrix_no_bc_for_dual = matrix_no_bc.copy()

        # * System Matrix = system_matrix + weight_mass_matrix * mass_matrix
        matrix_no_bc = matrix_no_bc + mass_matrix_no_bc

        print("begin vectorize mass matrix ...")
        rows_mass, cols_mass, values_mass = scipy.sparse.find(
            mass_matrix_no_bc[:n_dofs["time_step"], :n_dofs["time_step"]])
        print("end vectorize mass matrix ...")

        mass_matrix_up_right_no_bc = coo_matrix(
            (values_mass, (rows_mass, mass_matrix_no_bc.shape[1] - n_dofs["time_step"] + cols_mass)), shape=mass_matrix_no_bc.shape)

        # mass_matrix_up_right_no_bc = np.zeros(
        #     (mass_matrix_no_bc.shape[0], mass_matrix_no_bc.shape[1]))

        # mass_matrix_up_right_no_bc[:n_dofs["time_step"], -n_dofs["time_step"]
        #     :] = mass_matrix_no_bc[:n_dofs["time_step"], :n_dofs["time_step"]].toarray()
        # mass_matrix_up_right_no_bc = scipy.sparse.csr_matrix(
        #     mass_matrix_up_right_no_bc)

        # plot sparsity pattern of mass matrix
        # plt.spy(mass_matrix_up_right_no_bc)
        # plt.show()
        # plt.spy(test_matrix)
        # plt.show()

        # print(
        #     f"Check if same: {np.linalg.norm(mass_matrix_up_right_no_bc.toarray() - test_matrix.toarray())}")

        # * lower diagonal
        # mass_matrix_down_right_no_bc = np.zeros(
        #     (mass_matrix_no_bc.shape[0], mass_matrix_no_bc.shape[1]))
        # mass_matrix_down_right_no_bc[-n_dofs["time_step"]:, -n_dofs["time_step"]
        #     :] = mass_matrix_no_bc[:n_dofs["time_step"], :n_dofs["time_step"]].toarray()
        # mass_matrix_down_right_no_bc = scipy.sparse.csr_matrix(
        #     mass_matrix_down_right_no_bc)

        # * lower diagonal
        mass_matrix_down_right_no_bc = coo_matrix(
            (values_mass, (mass_matrix_no_bc.shape[1] - n_dofs["time_step"] + rows_mass, mass_matrix_no_bc.shape[1] - n_dofs["time_step"] + cols_mass)), shape=mass_matrix_no_bc.shape)

        # * left down corner --> transposed for dual LES
        # mass_matrix_down_left_no_bc = np.zeros(
        #     (mass_matrix_no_bc.shape[0], mass_matrix_no_bc.shape[1]))
        # mass_matrix_down_left_no_bc[-n_dofs["time_step"]:, :n_dofs["time_step"]
        #                             ] = mass_matrix_no_bc[:n_dofs["time_step"], :n_dofs["time_step"]].toarray().T
        # mass_matrix_down_left_no_bc = scipy.sparse.csr_matrix(
        #     mass_matrix_down_left_no_bc)

        # * left down corner --> transposed for dual LES
        mass_matrix_down_left_no_bc = coo_matrix(
            (values_mass, (mass_matrix_no_bc.shape[1] - n_dofs["time_step"] + rows_mass, cols_mass)), shape=mass_matrix_no_bc.shape)

        # ? Apply mass_matrix to last time step? Thus diag(mass_matrix) = [0, 0, M.T] instead of [M.T, 0, 0]
        dual_matrix_no_bc = matrix_no_bc_for_dual.T + mass_matrix_no_bc.T

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
        initial_solution = np.loadtxt(
            OUTPUT_PATH + cycle + "/initial_solution.txt")

        # %% Primal FOM solve ------------------------------------------------------------------------
        # ! Benchmarked in Paraview against deal.ii solution --> minor errors thus assuming that solver is correct

        SKIP_PRIMAL = False

        # slab_properties["n_total"] = 100

        if not SKIP_PRIMAL:
            start_time = time.time()

            sparse_lu = scipy.sparse.linalg.splu(primal_matrix)

            primal_solutions_slab = {"value": [], "time": []}
            for i in range(slab_properties["n_total"]):
                if i == 0:
                    primal_rhs = primal_system_rhs[i].copy(
                    ) + mass_matrix_up_right.dot(np.zeros(primal_matrix.shape[0]))
                else:
                    primal_rhs = primal_system_rhs[i].copy(
                    ) + mass_matrix_up_right.dot(primal_solutions_slab["value"][-1])

                # primal_solution = scipy.sparse.linalg.spsolve(
                #     primal_matrix, primal_rhs)
                primal_solution = sparse_lu.solve(primal_rhs)
                # scipy.sparse.linalg.spsolve(
                # sparse_lu, primal_rhs)

                primal_solutions_slab["value"].append(primal_solution)
                primal_solutions_slab["time"].append(
                    slab_properties["time_points"][i])

            time_FOM = time.time() - start_time

            print(f"cycle={space_cycle}-{time_cycle} solved in {time_FOM:.2f}s")
            with open(LOG_PATH + "discretization.log", 'a') as f:
                f.write(f"cycle={space_cycle}-{time_cycle} solved in {time_FOM:.2f}s\n")

            # print("Primal FOM time:   " + str(time_FOM))
            print("n_dofs[space]:     ", n_dofs["space"])
            print("time steps:        ", slab_properties["n_total"])

            primal_solutions = {
                "value": [np.zeros(int(n_dofs["time_step"]))], "time": [0.]}

            for i, primal_solution_slab in enumerate(primal_solutions_slab["value"]):
                for j in range(len(slab_properties["time_points"][i][1:])):
                    range_start = int((j)*n_dofs["time_step"])
                    range_end = int((j+1)*n_dofs["time_step"])
                    primal_solutions["value"].append(
                        primal_solutions_slab["value"][i][range_start:range_end])
                    primal_solutions["time"].append(
                        slab_properties["time_points"][i][j+1])

            # for i, primal_solution in enumerate(primal_solutions["value"]):
            #     save_vtk(SAVE_PATH + f"/py_solution{i:05}.vtk", {"displacement": dof_matrix.dot(primal_solution[0:n_dofs["space"]]), "velocity": dof_matrix.dot(
            #         primal_solution[n_dofs["space"]:2 * n_dofs["space"]])}, grid, cycle=i, time=primal_solutions["time"][i])

            J_h_t = np.empty([slab_properties["n_total"], 1])
            for i in range(slab_properties["n_total"]):
                J_h_t[i] = primal_solutions_slab["value"][i].dot(
                    dual_rhs_no_bc[i])
            J_h[space_cycle, time_cycle] = np.sum(J_h_t)
            print("Cost functional:   " + str(J_h[space_cycle, time_cycle]))
            # save costfunctional to file
            np.savetxt("cf/" + cycle + ".txt", J_h_t)

            if time_cycle == 0:
                space_label.append(n_dofs["space"])
            if space_cycle == 0:
                time_label.append(n_dofs["time"])


J_h_rel = np.abs((J_h - J_h[-1, -1])/J_h[-1, -1]*100)

# print(J_h_rel)
print(f"BEST CF: {J_h[-1, -1]}")


table = tabulate(J_h, headers=time_label,
                 showindex=space_label, tablefmt="latex")
print(table)


table = tabulate(J_h_rel, headers=time_label,
                 showindex=space_label, tablefmt="latex")
print(table)

# Write table to file
with open(LOG_PATH + "table_discretization.txt", 'w') as f:
    f.write(table)
