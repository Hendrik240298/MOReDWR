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
from iPOD import reduce_vector, project_vector
# %% Vtk plotting
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

# %% save solutions to txt in scientific notation
def save_solution_txt(file_name, solutions):
    # save solution to filename in txt format
    for i, solution in enumerate(solutions["value"]):
         np.savetxt(file_name + f"{i:05}.txt", solution, fmt="%.16e")  
    np.savetxt(file_name + "_time.txt", solutions["time"], fmt="%.16e")   
     

# %% read in solution that is saved in save_solution_txt
def load_solution_txt(file_name):

    solutions = {}
    solutions["time"] = np.loadtxt(file_name + "_time.txt")
    solutions["value"] = []
    for i in range(len(solutions["time"])):
        solutions["value"].append(np.loadtxt(file_name + f"{i:05}.txt"))
    # read in solution from filename in txt format
    return solutions

#%% read_in_discretization
def read_in_discretization(PATH): 
    # creating grid for vtk output
    grid = []
    with open(PATH + "/solution00000.vtk", "r") as f:
        writing = False
        for line in f:
            if (not writing and line.startswith("POINTS")):
                writing = True
            if writing:
                grid.append(line.strip("\n"))
                if line.startswith("POINT_DATA"):
                    break
    # creating dof_matrix for vtk output
    dof_vector = np.loadtxt(PATH + "/dof.txt").astype(int)
    dof_matrix = scipy.sparse.dok_matrix((dof_vector.shape[0], dof_vector.max()+1))
    for i, j in enumerate(list(dof_vector)):
        dof_matrix[i, j] = 1.


    # Definition coordinates
    coordinates_x = np.loadtxt(PATH + "/coordinates_x.txt")
    list_coordinates_t = []
    for f in sorted([f for f in os.listdir(
            PATH) if "coordinates_t" in f]):
        list_coordinates_t.append(np.loadtxt(PATH + "/" + f))
        
    slab_properties = {"n_total": len(list_coordinates_t)}
    coordinates_t = np.hstack(list_coordinates_t)
    coordinates = np.vstack((
        np.tensordot(coordinates_t, np.ones_like(coordinates_x), 0).flatten(),
        np.tensordot(np.ones_like(coordinates_t), coordinates_x, 0).flatten()
    )).T

    index2measuredisp = np.where((coordinates_x.T == (6, 0.5, 0)).all(axis=1))[0][1]
    print(f"index: {index2measuredisp}")

    # n_dofs["space"] per unknown u and v -> 2*n_dofs["space"] for one block
    n_dofs = {"space": coordinates_x.shape[1], "time": coordinates_t.shape[0]}
    n_dofs["solperstep"] =  int(n_dofs["time"] / slab_properties["n_total"] - 1)
    n_dofs["time_step"] = 2*n_dofs["space"] # dofs per submatrix of each time quadrature point

   
    slab_properties["ordering"] =  np.argsort(list_coordinates_t[0]).astype(int)   # ordering of time quadrature points on slab wrt time
    slab_properties["n_time_unknowns"] =  len(slab_properties["ordering"][1:])     # number of time steps with unknowns: e.g. in cg its len(ordering) -1 
    slab_properties["time_points"] = [list_coordinates_t[i][slab_properties["ordering"]] for i in range(slab_properties["n_total"])] # time points of each slab
    
    return n_dofs, slab_properties, index2measuredisp, dof_matrix, grid

# %% read_in_LES
def read_in_LES(OUTPUT_PATH, matrix_name, rhs_name):
    # Reading in system matix matrix_no_bc
    [data, row, column] = np.loadtxt(OUTPUT_PATH + matrix_name) #"/matrix_no_bc.txt")
    matrix_no_bc = scipy.sparse.csr_matrix(
        (data, (row.astype(int), column.astype(int))))
    
    # Reading in rhs wo  bc
    rhs_no_bc = []
    for f in sorted([f for f in os.listdir(OUTPUT_PATH) 
                     if rhs_name in f]):
        rhs_no_bc.append(np.loadtxt(OUTPUT_PATH + "/" + f))
    
    return matrix_no_bc, rhs_no_bc


# %% apply_boundary_conditions
def apply_boundary_conditions(matrix_no_bc, rhs_no_bc, path_boundary_ids):

    boundary_ids = np.loadtxt(path_boundary_ids).astype(int)

    # Enforcing BC to primal matrix
    primal_matrix = matrix_no_bc.tocsr()
    primal_system_rhs = []
    for row in boundary_ids:
        for col in primal_matrix.getrow(row).nonzero()[1]:
            primal_matrix[row, col] = 1. if row == col else 0.

    # Enforcing BC to primah rhs
    for rhs_no_bc_sample in rhs_no_bc:
        primal_system_rhs.append(rhs_no_bc_sample)
        for row in boundary_ids:
            primal_system_rhs[-1][row] = 0.0

    return primal_matrix, primal_system_rhs
    
# %% reorder_matrix
def reorder_matrix(matrix, slab_properties, n_dofs):
    matrix_ordered =  matrix.copy() #np.zeros_like(matrix_no_bc)

    for i in range(len(slab_properties["ordering"])):
        #note: col=row and vice versa
        col_new_low = i*n_dofs["time_step"]
        col_new_up = (i+1)*n_dofs["time_step"]
        col_old_low = slab_properties["ordering"][i]*n_dofs["time_step"]
        col_old_up = (slab_properties["ordering"][i]+1)*n_dofs["time_step"]
        # print(f"{col_old_lsow}, {col_old_up} --> {col_new_low}, {col_new_up} ")
        
        for j in range(len(slab_properties["ordering"])):
            row_new_low = j*n_dofs["time_step"]
            row_new_up = (j+1)*n_dofs["time_step"]
            row_old_low = slab_properties["ordering"][j]*n_dofs["time_step"]
            row_old_up = (slab_properties["ordering"][j]+1)*n_dofs["time_step"]
            # print(f"[{col_old_low}:{col_old_up}, {row_old_low}:{row_old_up}] --> [{col_new_low}:{col_new_up}, {row_new_low}:{row_new_up}]")

            # primal matricies
            matrix_ordered[col_new_low:col_new_up,row_new_low:row_new_up] \
                = matrix[col_old_low:col_old_up,row_old_low:row_old_up]
                
    return matrix_ordered
    
# %% reorder_vector
def reorder_vector(vector, slab_properties, n_dofs):
    vector_ordered = vector.copy()
    for i in range(len(slab_properties["ordering"])):
        index_new_low = i*n_dofs["time_step"]
        index_new_up = (i+1)*n_dofs["time_step"]
        index_old_low = slab_properties["ordering"][i]*n_dofs["time_step"]
        index_old_up = (slab_properties["ordering"][i]+1)*n_dofs["time_step"]

        vector_ordered[index_new_low:index_new_up] = vector[index_old_low:index_old_up].copy()
        
    return vector_ordered
# %% solve_primal_FOM_step
def solve_primal_FOM_step(primal_solutions, D_wbc, C_wbc, primal_system_rhs, slab_properties, n_dofs, i):
    primal_rhs = primal_system_rhs[n_dofs["time_step"]:].copy() - C_wbc.dot(primal_solutions["value"][-1])
    
    primal_solution = scipy.sparse.linalg.spsolve(D_wbc, primal_rhs)
    for j in range(slab_properties["n_time_unknowns"]):
        primal_solutions["value"].append(primal_solution[j*n_dofs["time_step"]:(j+1)*n_dofs["time_step"]])
        primal_solutions["time"].append(slab_properties["time_points"][i][j+1])
    
    return primal_solutions
    
# %% solve_dual_FOM_step
def solve_dual_FOM_step(dual_solutions, A_wbc, B_wbc, dual_system_rhs, slab_properties, n_dofs, i):
    dual_rhs = dual_system_rhs[:-n_dofs["time_step"]].copy() - B_wbc.dot(dual_solutions["value"][-1])
    dual_solution = scipy.sparse.linalg.spsolve(A_wbc, dual_rhs)

    for j in list(range(slab_properties["n_time_unknowns"]))[::-1]:
        dual_solutions["value"].append(dual_solution[j*n_dofs["time_step"]:(j+1)*n_dofs["time_step"]])
        dual_solutions["time"].append(slab_properties["time_points"][i][j])

    return dual_solutions

# %% solve_primal_ROM_step
def solve_primal_ROM_step(projected_reduced_solutions, reduced_solution_old, D_reduced, C_reduced, rhs_no_bc, pod_basis, slab_properties, n_dofs, i):

    reduced_rhs = reduce_vector(rhs_no_bc[n_dofs["time_step"]:].copy(), pod_basis) - C_reduced.dot(reduced_solution_old)
    reduced_solution = np.linalg.solve(D_reduced, reduced_rhs)
    
    reduced_solution_old = reduced_solution[(slab_properties["n_time_unknowns"]-1)*n_dofs["reduced_primal"]:(slab_properties["n_time_unknowns"])*n_dofs["reduced_primal"]]

    for j in range(slab_properties["n_time_unknowns"]):
        projected_reduced_solutions["value"].append(project_vector(reduced_solution[j*n_dofs["reduced_primal"]:(j+1)*n_dofs["reduced_primal"]],pod_basis))
        projected_reduced_solutions["time"].append(slab_properties["time_points"][i][j+1])

    return projected_reduced_solutions, reduced_solution_old
    
# %% compute error estimate

def error_estimator(projected_reduced_solutions, dual_projected_reduced_solutions, matrix_no_bc, rhs_no_bc, slab_properties):

    projected_slab = { "value": [], "time": [] }
    dual_projected_slab = { "value": [], "time": [] }

    projected_slab["value"] = np.hstack(projected_reduced_solutions["value"][-slab_properties["n_time_unknowns"]-1:])
    projected_slab["time"] = np.hstack(projected_reduced_solutions["time"][-slab_properties["n_time_unknowns"]-1:])


    # find argument where time of dual_projected_reduced_solutions is equal to time of projected_slab
    index_of_dual = []
    for j in range(len(projected_slab["time"])):
        for i in range(len(dual_projected_reduced_solutions["time"])):
            if dual_projected_reduced_solutions["time"][i] == projected_slab["time"][j]:
                index_of_dual.append(i)
                break

    # hstack last entries of projected_reduced_solutions to obtain slab
    dual_projected_slab["value"] = np.hstack([dual_projected_reduced_solutions["value"][i] for i in index_of_dual])

    # hstack i-th last entries of projected_reduced_solutions["time"] to obtain slab
    dual_projected_slab["time"] = np.hstack([dual_projected_reduced_solutions["time"][i] for i in index_of_dual])

    residual_slab = - matrix_no_bc.dot(projected_slab["value"]) + rhs_no_bc

    error_estimate = np.abs(np.dot(dual_projected_slab["value"], residual_slab))

    return error_estimate    


def evaluate_cost_functional(projected_reduced_solutions, dual_rhs_no_bc, slab_properties,i):

    time_points = slab_properties["time_points"][i]

    # find arguments where time of projected_reduced_solutions is equal to time_points
    index_of_primal = []
    for j in range(len(time_points)):
        for k in list(range(len(projected_reduced_solutions["time"])))[::-1]:
            if projected_reduced_solutions["time"][k] == time_points[j]:
                index_of_primal.append(k)
                break

    primal_projected_slab = np.hstack([projected_reduced_solutions["value"][i] for i in index_of_primal])

    cost_functional = np.dot(primal_projected_slab, dual_rhs_no_bc)

    return cost_functional
