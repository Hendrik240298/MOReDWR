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
from auxiliaries import save_vtk, read_in_LES, apply_boundary_conditions, read_in_discretization,solve_primal_FOM_step, solve_dual_FOM_step, solve_primal_ROM_step, reorder_matrix,reorder_vector
#import imageio

PLOTTING = False
MOTHER_PATH = "/home/hendrik/Code/MORe_DWR/Elastodynamics/"
OUTPUT_PATH = MOTHER_PATH + "/Data/3D/Rod/"
OUTPUT_PATH_DUAL = MOTHER_PATH + "Dual_Elastodynamics/Data/3D/Rod/"
cycle = "cycle=1"
SAVE_PATH = MOTHER_PATH + "Data/ROM/" + cycle + "/"

LOAD_SOLUTION = False

print(f"\n{'-'*12}\n| {cycle}: |\n{'-'*12}\n")

# SAVE_PATH = cycle + "/output_ROM/"

#"../../FOM/slabwise/output_" + CASE + "/dim=1/"

# ENERGY_PRIMAL_DISPLACEMENT = 0.99999999 
# ENERGY_PRIMAL_VELOCITY = 0.9999999
ENERGY_DUAL = 0.999999

ENERGY_PRIMAL = {"displacement": 0.99999999, \
                 "velocity":     0.99999999}

# %% read in properties connected to discretization
n_dofs, slab_properties, index2measuredisp, dof_matrix, grid = read_in_discretization(OUTPUT_PATH + cycle)

# %% Reading in matricies and rhs without bc
# matrix_no_bc, rhs_no_bc, dual_matrix_no_bc, dual_rhs_no_bc = read_in_LES(OUTPUT_PATH, OUTPUT_PATH_DUAL, cycle)
matrix_no_bc, rhs_no_bc = read_in_LES(OUTPUT_PATH + cycle, "/matrix_no_bc.txt", "primal_rhs_no_bc")
dual_matrix_no_bc, dual_rhs_no_bc = read_in_LES(OUTPUT_PATH_DUAL + cycle, "/dual_matrix_no_bc.txt", "dual_rhs_no_bc")
# %% Enforcing BC to primal und dual systems 
# primal_matrix, primal_system_rhs, dual_matrix, dual_system_rhs = apply_boundary_conditions(matrix_no_bc, rhs_no_bc, dual_matrix_no_bc, dual_rhs_no_bc, OUTPUT_PATH + cycle + "/boundary_id.txt")
primal_matrix, primal_system_rhs = apply_boundary_conditions(matrix_no_bc, rhs_no_bc, OUTPUT_PATH + cycle + "/boundary_id.txt")
dual_matrix, dual_system_rhs = apply_boundary_conditions(dual_matrix_no_bc, dual_rhs_no_bc, OUTPUT_PATH + cycle + "/boundary_id.txt")

# %% read in IC
initial_solution = np.loadtxt(OUTPUT_PATH + cycle + "/initial_solution.txt")

# %% Definition for dofs variables and slab properties

# find the index for coordinates (6,0.5,0)

# ordering the quardature points on slab wrt time
#for i in range(slab_properties["n_total"]):
    


# time_step_size = list_coordinates_t[0][slab_properties["ordering"][1]]-list_coordinates_t[0][slab_properties["ordering"][0]]
# solution_times = [list_coordinates_t[0][0]]
# for i in range(slab_properties["n_total"]):
#     for j in (slab_properties["ordering"][1:]-1):
#         solution_times.append(list_coordinates_t[i][j])
# ------------
    
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
        
# %% Definition sub system 

# PRIMAL 
# --------------    
#  ~  |    ~        x_1
# --------------
#     |             x_2
#  C  |    D        ...
#     |             x_n
# --------------


# Dual 
# --------------
#        |           z_1
#    A   |  B        ...
#        |           z_n-1
# -------------- 
#    ~   |  ~        z_n
# --------------

# DEAL with dumb deal.ii ordering of time_steps
        
# dual problem matricies
A = dual_matrix_no_bc[:-n_dofs["time_step"], :-n_dofs["time_step"]]
B = dual_matrix_no_bc[:-n_dofs["time_step"], -n_dofs["time_step"]:]

A_wbc = dual_matrix[:-n_dofs["time_step"], :-n_dofs["time_step"]]
B_wbc = dual_matrix[:-n_dofs["time_step"], -n_dofs["time_step"]:]


# primal problem matricies
C = matrix_no_bc[n_dofs["time_step"]:, :n_dofs["time_step"]]
D = matrix_no_bc[n_dofs["time_step"]:, n_dofs["time_step"]:]

C_wbc = primal_matrix[n_dofs["time_step"]:, :n_dofs["time_step"]]
D_wbc = primal_matrix[n_dofs["time_step"]:, n_dofs["time_step"]:]



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
    # creating primal rhs and applying BC to it
    primal_solutions = solve_primal_FOM_step(primal_solutions, D_wbc, C_wbc, primal_system_rhs[i], slab_properties, n_dofs, i)

end_execution = time.time()

execution_time_FOM = end_execution - start_execution

print("Primal FOM time:   " + str(execution_time_FOM))
print("n_dofs[space] =", n_dofs["space"])


for i, primal_solution in enumerate(primal_solutions["value"]):
    save_vtk(SAVE_PATH + f"/py_solution{i:05}.vtk", {"displacement": dof_matrix.dot(primal_solution[0:n_dofs["space"]]), "velocity": dof_matrix.dot(
        primal_solution[n_dofs["space"]:2 * n_dofs["space"]])}, grid, cycle=i, time=primal_solutions["time"][i])

primal_solutions_slab =  {"value": [], "time": []}
for i in range(slab_properties["n_total"]):
    primal_solutions_slab["value"].append(np.hstack( primal_solutions["value"][i*(n_dofs['solperstep']): (i+1)*(n_dofs['solperstep'])+1]).T)
    primal_solutions_slab["time"].append(slab_properties["time_points"][i])


# %% dual FOM desperation solve

# start_execution = time.time()
# last_dual_solution = np.zeros((n_dofs["time_step"],))

# dual_solutions = [np.zeros((n_dofs["time_step"],))]

# dual_matrix_full_solve = dual_matrix.copy()

# for row in range(n_dofs["time_step"]):
#     for col in dual_matrix_full_solve.getrow(row).nonzero()[1]:
#         dual_matrix_full_solve[n_dofs['solperstep']*n_dofs["time_step"]+row,col] = 1. if n_dofs['solperstep']*n_dofs["time_step"]+row == col else 0.

# print("assembled full dual matrix")

# for i in list(range(slab_properties["n_total"]))[::-1]:
    
#     dual_rhs = np.hstack((dual_system_rhs[i][:-n_dofs["time_step"]].copy(),last_dual_solution))
#     dual_solution = scipy.sparse.linalg.spsolve(dual_matrix_full_solve, dual_rhs)
    
#     for j in list(range(slab_properties["n_time_unknowns"]))[::-1]:
#         dual_solutions.append(dual_solution[j*n_dofs["time_step"]:(j+1)*n_dofs["time_step"]])

#     last_dual_solution = dual_solutions[-1]
# end_execution = time.time()

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
    dual_solutions = []
    for f in sorted([f for f in os.listdir(OUTPUT_PATH_DUAL + cycle) if "dual_solution" in f]):
        tmp_sol = np.loadtxt(OUTPUT_PATH_DUAL + cycle + "/" + f)
        for j in slab_properties["ordering"][slab_properties["ordering"] < n_dofs["solperstep"]]:
            dual_solutions.append(tmp_sol[j*n_dofs["time_step"]:(j+1)*n_dofs["time_step"]])
    # final condition = 0
    dual_solutions.append(np.zeros((n_dofs["time_step"],)))
        
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


# %% Definition ROM ingredients

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


for primal_solution in primal_solutions["value"]:#[0:10]:
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
# needed for dual
# A_reduced = reduce_matrix(A, pod_basis_dual, pod_basis_dual)
# B_reduced = reduce_matrix(B, pod_basis_dual, pod_basis_dual)
# J_1_reduced = reduce_matrix(J_1, pod_basis_dual, pod_basis)
# J_2_reduced = reduce_matrix(J_1, pod_basis_dual, pod_basis)
# needed for primal
C_reduced = reduce_matrix(C, pod_basis, pod_basis)
D_reduced = reduce_matrix(D, pod_basis, pod_basis)

# %% Primal ROM solve
# reduced_solutions = {"value": [], "time": []}

reduced_solutions = []
reduced_solution_old = reduce_vector(initial_solution[:], pod_basis)

# reduced_dual_solutions = []
# reduced_dual_solution_old = reduce_vector(dual_solutions[0], pod_basis_dual)
# forward_reduced_dual_solution = np.zeros_like(reduce_vector(dual_solutions[0], pod_basis_dual))

projected_reduced_solutions = {"value": [project_vector(reduce_vector(initial_solution[:], pod_basis), pod_basis)], 
                               "time": [0.]}

# projected_reduced_solutions = [project_vector(reduce_vector(initial_solution[:], pod_basis), pod_basis)]

last_projected_reduced_solution = projected_reduced_solutions
projected_reduced_solutions_before_enrichment = []

# projected_reduced_dual_solutions = [project_vector(reduce_vector(dual_solutions[0], pod_basis_dual), pod_basis_dual)]

# dual_residual = []
# dual_residual.append(0)

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
    
    # build vectors for estimator --> on large matrix
    projected_reduced_solutions_vec_shorti = np.hstack(projected_reduced_solutions["value"][-slab_properties["n_time_unknowns"]:])
    projected_reduced_solutions_vec_shorti_old = projected_reduced_solutions["value"][-slab_properties["n_time_unknowns"]-1]
    residual_shorti = - D.dot(projected_reduced_solutions_vec_shorti) - C.dot(projected_reduced_solutions_vec_shorti_old) \
                    + rhs_no_bc[i][n_dofs["time_step"]:]
    
    temporal_interval_error_shorti.append(np.abs(np.dot( residual_shorti, dual_solutions_slab["value"][i][n_dofs["time_step"]:] ))) 
       
    projected_reduced_solutions_vec = np.hstack(projected_reduced_solutions["value"][-slab_properties["n_time_unknowns"]-1:])
    residual = - matrix_no_bc.dot(projected_reduced_solutions_vec) + rhs_no_bc[i].copy()
    residual[:n_dofs["time_step"]] = 0.
    # dual_solution = np.hstack(dual_solutions[slab_properties["n_time_unknowns"]*i:slab_properties["n_time_unknowns"]*(i+1)+1])
    
    temporal_interval_error.append(np.abs(np.dot(dual_solutions_slab["value"][i], residual)))
    # eta_rel = eta/(eta + J(u_r)) ~ eta/J(u_h)
    temporal_interval_error_relative.append(temporal_interval_error[-1] / \
                            np.abs(temporal_interval_error[-1] + np.dot(projected_reduced_solutions_vec,dual_rhs_no_bc[i])))
    # error indicator
    # temporal_interval_error_relative.append( 1.0 if i % 5 == 0 else 0.0 )

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
        # print(f"len of ps after update : {len(projected_reduced_solutions[:-slab_properties["n_time_unknowns"]])}")
        # print(f"len sol after update  : {len(projected_reduced_solutions)}")

        n_dofs["reduced_primal"] = pod_basis["displacement"].shape[1] + pod_basis["velocity"].shape[1]
        reduced_solution_old = reduce_vector(projected_reduced_solutions["value"][-1],pod_basis)
    # projected_reduced_solutions.append(
    #     project_vector(reduced_solution, pod_basis))

    # reduced_dual_rhs=J_2_reduced.dot(reduced_solution) + J_1_reduced.dot(reduced_solution_old)
    # reduced_dual_rhs -= B_reduced.T.dot(forward_reduced_dual_solutions[-1])

    # reduced_dual_solution=np.linalg.solve(A_reduced.T, reduced_dual_rhs)

    # ATTENTION WE HAVE TO SPLIT THE REDUCED SOLUTION SINCE IT CONTAINS NOW MORE THEN ONE TIMESTEP
    # IF higher than cg(1)
    
    last_projected_reduced_solution = projected_reduced_solutions["value"][-1]

end_execution = time.time()
execution_time_ROM = end_execution - start_execution
print("ROM time:        " + str(execution_time_ROM))


for i, projected_reduced_solution in enumerate(projected_reduced_solutions["value"]):
    save_vtk(SAVE_PATH + f"/projected_solution{i:05}.vtk", {"displacement": dof_matrix.dot(projected_reduced_solution[0:n_dofs["space"]]), "velocity": dof_matrix.dot(
        projected_reduced_solution[n_dofs["space"]:2 * n_dofs["space"]])}, grid, cycle=i, time=projected_reduced_solutions["time"][i])

projected_reduced_solutions_slab =  {"value": [], "time": []}
for i in range(slab_properties["n_total"]):
    projected_reduced_solutions_slab["value"].append(np.hstack( projected_reduced_solutions["value"][i*(n_dofs['solperstep']): (i+1)*(n_dofs['solperstep'])+1]).T)
    projected_reduced_solutions_slab["time"].append(slab_properties["time_points"][i])

# %% Computing Cost functional

J_r_t = np.empty([slab_properties["n_total"], 1])
J_h_t = np.empty([slab_properties["n_total"], 1])   
# TODO: is this correct?
for i in range(slab_properties["n_total"]):
    
    # tmp_dual_rhs = []
    # for j in slab_properties["ordering"]:
    #     tmp_dual_rhs.append(dual_rhs[j*n_dofs["time_step"]:(j+1)*n_dofs["time_step"]])
    
    J_r_t[i] = projected_reduced_solutions_slab["value"][i].dot(dual_rhs_no_bc[i])
    J_h_t[i] = primal_solutions_slab["value"][i].dot(dual_rhs_no_bc[i])
    # for j in range(n_dofs["solperstep"]+1):
    #    J_r_t[i] += projected_reduced_solutions[i*n_dofs["solperstep"]+j].dot(tmp_dual_rhs[j])

# J_h_t = np.empty([slab_properties["n_total"], 1])      
        
# # TODO: Is the computation of J_h_t correct?
# for i, dual_rhs in enumerate(dual_rhs_no_bc):
    
#     tmp_dual_rhs = []
#     for j in slab_properties["ordering"]:
#         tmp_dual_rhs.append(dual_rhs[j*n_dofs["time_step"]:(j+1)*n_dofs["time_step"]])
    
#     J_h_t[i] = 0.
#     for j in range(n_dofs["solperstep"]+1):
#        J_h_t[i] += primal_solutions[i*n_dofs["solperstep"]+j].dot(tmp_dual_rhs[j])




# for i, projected_reduced_solution in enumerate(projected_reduced_solutions):
#     save_vtk(SAVE_PATH + f"/primal_solution{i:05}.vtk",
#              {"displacement": dof_matrix.dot(projected_reduced_solution[0:n_dofs["space"]]), "velocity":    dof_matrix.dot(projected_reduced_solution[n_dofs["space"]:2 * n_dofs["space"]])}, grid, cycle=i, time=(list_coordinates_t[i-1][1] if i > 0 else 0.))

# for i, projected_reduced_dual_solution in enumerate(projected_reduced_dual_solutions):
#     save_vtk(OUTPUT_PATH + cycle + f"/dual_solution{i:05}.vtk",
#              {"displacement": dof_matrix.dot(projected_reduced_dual_solution[0:n_dofs["space"]]), "velocity":    dof_matrix.dot(projected_reduced_dual_solution[n_dofs["space"]:2 * n_dofs["space"]])}, grid, cycle=i, time=(list_coordinates_t[i-1][1] if i > 0 else 0.))

    # save_vtk(OUTPUT_PATH + cycle + f"/py_solution{i:05}.vtk",
# 	save_vtk(OUTPUT_PATH + cycle + f"/py_solution{i+1:05}.vtk", {"displacement": dof_matrix.dot(primal_solution[0:n_dofs["space"]]), \
#             "velocity": dof_matrix.dot(primal_solution[n_dofs["space"]:2 * n_dofs["space"]])}, grid, cycle=i+1, time=list_coordinates_t[i][1])
error_primal_displacement = []
# error_dual_displacement = []
error_primal_velo = []
error_primal_displacement_pointwise = []
# error_dual_velo = []
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
    # if np.linalg.norm(dual_solutions[i][n_dofs["space"]:]) != 0.0 and np.linalg.norm(dual_solutions[i][:n_dofs["space"]]) != 0.0:
    #     error_dual_displacement.append(np.linalg.norm(dual_solutions[i][:n_dofs["space"]] - projected_reduced_dual_solutions[i][:n_dofs["space"]])/ np.linalg.norm(projected_reduced_dual_solutions[i][:n_dofs["space"]]) )
    #     error_dual_velo.append(np.linalg.norm(dual_solutions[i][n_dofs["space"]:] - projected_reduced_dual_solutions[i][n_dofs["space"]:])/np.linalg.norm(dual_solutions[i][n_dofs["space"]:]) )
    # else:
    #     error_dual_displacement.append(0)
    #     error_dual_velo.append(0)

# %% Plotting

time_step_size = 40.0 / (slab_properties["n_total"])

# # plot sigs
# plt.rc('text', usetex=True)
# # plt.rcParams["figure.figsize"] = (10,2)
# plt.plot(np.arange(0, pod_basis["displacement"].shape[1]),
#          singular_values["displacement"], label="displacement")
# plt.plot(np.arange(0, pod_basis["velocity"].shape[1]),
#          singular_values["velocity"], label="velocity")
# # plt.plot(np.arange(0, pod_basis_dual["displacement"].shape[1]),
# #          singular_values_dual["displacement"], label="displacement_dual")
# # plt.plot(np.arange(0, pod_basis_dual["velocity"].shape[1]),
# #          singular_values_dual["velocity"], label="velocity_dual")
# plt.grid()
# plt.yscale('log')
# plt.legend()
# # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
# plt.show()


# # plot norm of displace,mnet
# plt.rc('text', usetex=True)
# # plt.rcParams["figure.figsize"] = (10,2)
# plt.plot(np.arange(0, slab_properties["n_total"]+1),
#          error_primal_displacement, label="primal")
# # plt.plot(np.arange(0, slab_properties["n_total"]+1),
# #          error_dual_displacement, label="dual")
# plt.yscale('log')
# plt.legend()
# # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
# plt.show()

# # plot norm of velocity
# plt.rc('text', usetex=True)
# # plt.rcParams["figure.figsize"] = (10,2)
# plt.plot(np.arange(0, slab_properties["n_total"]+1),
#          error_primal_velo, label="norm error displacement")
# # plt.plot(np.arange(0, slab_properties["n_total"]+1),
# #          error_dual_velo, label="dual")
# # plt.yscale('log')
# plt.legend()
# # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
# plt.show()

# # plot norm of velocity
# plt.rc('text', usetex=True)
# # plt.rcParams["figure.figsize"] = (10,2)
# plt.plot(np.arange(0, slab_properties["n_total"]+1),
#          error_primal_velo, label="norm error velocity")
# # plt.plot(np.arange(0, slab_properties["n_total"]+1),
# #          error_dual_velo, label="dual")
# # plt.yscale('log')
# plt.legend()
# # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
# plt.show()



# plot pointwise displacement at (6,0.5,0)
plt.rc('text', usetex=True)
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



# # plot rel error pointwise displacement at (6,0.5,0)
# plt.rc('text', usetex=True)
# # plt.rcParams["figure.figsize"] = (10,2)
# plt.plot(np.arange(0, slab_properties["n_total"]+1),
#          error_primal_displacement_pointwise, label="pw displacement")
# # plt.plot(np.arange(0, slab_properties["n_total"]+1),
# #          error_dual_displacement, label="dual")
# plt.yscale('log')
# plt.legend()
# # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
# plt.show()



# plot temporal evolution of cost funtiponal
plt.rc('text', usetex=True)
# plt.rcParams["figure.figsize"] = (10,2)
plt.plot(np.vstack(projected_reduced_solutions_slab["time"])[:,-1],
          J_h_t, color='r', label="$u_h$")
plt.plot(np.vstack(projected_reduced_solutions_slab["time"])[:,-1],
          J_r_t, '--', c='#1f77b4', label="$u_N$")
plt.grid()
plt.legend()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.xlabel('$t \; [$s$]$')
plt.ylabel("$J(u)\\raisebox{-.5ex}{$|$}_{Q_l}$")
plt.xlim([0, slab_properties["n_total"]*time_step_size])
#plt.title("temporal evaluation of cost funtional")

plt.show()

# %%
# plot temporal evolution of cost funtiponal
plt.rcParams["figure.figsize"] = (10,2)
plt.plot(np.vstack(projected_reduced_solutions_slab["time"])[:,-1],
          np.array(temporal_interval_error), c='#1f77b4', label="estimate")
plt.plot(np.vstack(projected_reduced_solutions_slab["time"])[:,-1],
          np.array(temporal_interval_error_shorti), c='black', label="estimate")
plt.plot(np.vstack(projected_reduced_solutions_slab["time"])[:,-1],
          np.abs(J_h_t-J_r_t), color='r', label="error")
plt.grid()
plt.legend()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.xlabel('$t \; [$s$]$')
plt.ylabel("$error$")
plt.xlim([0, slab_properties["n_total"]*time_step_size])

plt.show()
