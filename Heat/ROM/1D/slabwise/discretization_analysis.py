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
from tabulate import tabulate


PLOTTING = False
INTERPOLATION_TYPE = "cubic"  # "linear", "cubic"
CASE = ""  # "two" or "moving"
MOTHER_PATH = "/home/ifam/fischer/Code/MORe_DWR/Heat/"
MOTHER_PATH = "/home/hendrik/Code/MORe_DWR/Heat/"
OUTPUT_PATH = MOTHER_PATH + "Data/1D/moving_source/slabwise/FOM/"

space_cycles = 9
time_cycles = 7

space_label = []
time_label = []
# for time_cycle in range(time_cycles):
#     time_label.append(20 * 2**time_cycle)

J_h = np.zeros([space_cycles, time_cycles])
for time_cycle in range(time_cycles):
    for space_cycle in range(space_cycles):
        cycle = f"cycle={space_cycle}-{time_cycle}"

        # print(f"\n{'-'*12}\n| {cycle}: |\n{'-'*12}\n")
        # NO BC
        [data, row, column] = np.loadtxt(
            OUTPUT_PATH + cycle + "/matrix_no_bc.txt")
        matrix_no_bc = scipy.sparse.csr_matrix(
            (data, (row.astype(int), column.astype(int))))

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
            list_coordinates_t.append(
                np.loadtxt(OUTPUT_PATH + cycle + "/" + f))
        n_slabs = len(list_coordinates_t)
        coordinates_t = np.hstack(list_coordinates_t)
        coordinates = np.vstack((
            np.tensordot(coordinates_t, np.ones_like(
                coordinates_x), 0).flatten(),
            np.tensordot(np.ones_like(coordinates_t),
                         coordinates_x, 0).flatten()
        )).T
        # print(f"coordinates_x.shape: {coordinates_x.shape}")
        n_dofs = {
            "space": coordinates_x.shape[0], "time": coordinates_t.shape[0]}

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

        J_h_t = np.empty([n_slabs, 1])
        for i in range(n_slabs):
            J_h_t[i] = np.dot(primal_solutions[i],
                              mass_matrix_no_bc.dot(primal_solutions[i]))
        J_h[space_cycle, time_cycle] = np.sum(J_h_t)

        if time_cycle == 0:
            space_label.append(n_dofs["space"])
        if space_cycle == 0:
            time_label.append(n_dofs["time"])

# print(J_h)


J_h_rel = np.abs((J_h - J_h[-1, -1])/J_h[-1, -1]*100)

# print(J_h_rel)

print(f"BEST CF: {J_h[-1, -1]}")

table = tabulate(J_h_rel, headers=time_label,
                 showindex=space_label, tablefmt="latex")
print(table)
