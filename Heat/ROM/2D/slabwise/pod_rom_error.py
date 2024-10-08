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
from iPOD import iPOD

PLOTTING = True
INTERPOLATION_TYPE = "nearest"  # "linear", "cubic"
LOAD_PRIMAL_SOLUTION = False
CASE = ""  # "two" or "moving"
OUTPUT_PATH = "../../../Data/2D/rotating_circle/slabwise/"
cycle = "cycle=3"
SAVE_PATH = "../../../Data/2D/rotating_circle/slabwise/" + cycle + "/output_ROM/"
#"../../FOM/slabwise/output_" + CASE + "/dim=1/"



print(f"\n{'-'*12}\n| {cycle}: |\n{'-'*12}\n")
# NO BC
[data, row, column] = np.loadtxt(OUTPUT_PATH + cycle + "/matrix_no_bc.txt")
matrix_no_bc = scipy.sparse.csr_matrix(
    (data, (row.astype(int), column.astype(int))))
A = matrix_no_bc[:-1,:-1].toarray()
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

# %% applying BC to dual matrix
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

# %% POD init
total_energy = 0
POD = np.empty([0, 0])
bunch = np.empty([0, 0])
singular_values = np.empty([0, 0])

# ------------
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

    if LOAD_PRIMAL_SOLUTION:
        primal_solutions.append(
            np.loadtxt(
                OUTPUT_PATH +
                cycle +
                f"/solution_{(5-len(str(i)))*'0'}{i}.txt"))
    else:
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
    dual_rhs = dual_rhs_no_bc[i].copy()
    # dual_rhs = mass_matrix_no_bc.dot(primal_solutions[i])
    if True: #i%3 > 0:
        dual_rhs -= jump_matrix_no_bc.T.dot(last_dual_solution)
    for row in boundary_ids:
        dual_rhs[row] = 0.

    dual_solutions.append(
        scipy.sparse.linalg.spsolve(dual_matrix, dual_rhs))

    last_dual_solution = dual_solutions[-1]

# dual solution
dual_solutions = dual_solutions[::-1]
dual_solution = np.hstack(dual_solutions)

# %% goal functionals
J = {"u_h": 0., "u_r": []}
J_h_t = np.empty([n_slabs, 1])
for i in range(n_slabs):
    J_h_t[i] = np.dot(primal_solutions[i], dual_rhs_no_bc[i])
    J["u_h"] += np.dot(primal_solutions[i], dual_rhs_no_bc[i])#dual_rhs_no_bc[i])

colors = ["red", "blue", "green", "purple", "orange", "pink", "black"]
# ---------------- #
#%% PRIMAL POD-ROM #
# ---------------- #
print(f"#DoFs(space): {n_dofs['space']}")
print(f"#DoFs(time): {n_dofs['time']}")

# ---------------
# snapshot matrix
Y = primal_solution.reshape(n_dofs["time"], n_dofs["space"]).T
Y_dual = dual_solution.reshape(n_dofs["time"], n_dofs["space"]).T

dual_pod_basis, dual_eigen_values, _ = scipy.linalg.svd(
    Y_dual, full_matrices=False)
dual_eigen_values == np.sqrt(dual_eigen_values)

# desired energy ratio
ENERGY_RATIO_THRESHOLD = 0.9999
ENERGY_RATIO_THRESHOLD_DUAL = 0.9999

# determine number of POD basis vectors needed to preserve certain ratio
# of energy
r_dual = np.sum([(dual_eigen_values[:i].sum() / dual_eigen_values.sum()) <
                 ENERGY_RATIO_THRESHOLD_DUAL for i in range(n_dofs["time"])])

dual_pod_basis = dual_pod_basis[:, range(r_dual)]
dual_eigen_values = dual_eigen_values[range(r_dual)]

time_dofs_per_time_interval = int(n_dofs["time"] / n_slabs)
dofs_per_time_interval = time_dofs_per_time_interval * n_dofs["space"]

# %% ROM update


# def ROM_update(
#         pod_basis,
#         space_time_pod_basis,
#         reduced_system_matrix,
#         reduced_jump_matrix,
#         last_projected_reduced_solution):
#     # creating primal rhs and applying BC to it
    
#     extime_solve_FOM = 0.0
#     extime_iPOD = 0.0
#     extime_matrix = 0.0
    
#     start_time = time.time()
#     primal_rhs = rhs_no_bc[i].copy()
#     primal_rhs -= jump_matrix_no_bc.dot(last_projected_reduced_solution)
#     for row in boundary_ids:
#         primal_rhs[row] = 0.  # NOTE: hardcoding homogeneous Dirichlet BC

#     projected_reduced_solution = scipy.sparse.linalg.spsolve(
#         primal_matrix,  )
#     extime_solve_FOM = time.time() - start_time
    
#     start_time = time.time()
#     singular_values_tmp = singular_values
#     total_energy_tmp = total_energy
#     bunch_tmp = np.empty([0, 0])
#     if ((
#             projected_reduced_solution.shape[0] / n_dofs["space"]).is_integer()):
#         # onyl use first solution of slab since we assume that solutions are quite similar
#         for slab_step in range(
#                 # 1):
#                 int(projected_reduced_solution.shape[0] / n_dofs["space"])):
#             pod_basis, bunch_tmp, singular_values_tmp, total_energy_tmp = iPOD(pod_basis, bunch_tmp, singular_values_tmp, projected_reduced_solution[range(
#                 slab_step * n_dofs["space"], (slab_step + 1) * n_dofs["space"])], total_energy_tmp)
#     else:
#         print(
#             (projected_reduced_solution.shape[0] / n_dofs["space"]).is_integer())
#         print("Error building slapwise POD")
#     extime_iPOD = time.time() - start_time
    
#     start_time = time.time()
#     # change from the FOM to the POD basis
#     space_time_pod_basis = scipy.sparse.block_diag(
#         [pod_basis] * time_dofs_per_time_interval)
    
#     # start_time1 = time.time()
#     # reduced_system_matrix = space_time_pod_basis.T.dot(
#     #     matrix_no_bc.dot(space_time_pod_basis))
#     reduced_system_matrix = np.zeros([pod_basis.shape[1]*2,pod_basis.shape[1]*2])
#     start_time1_1 = time.time()
#     reduced_system_matrix[:pod_basis.shape[1],   :pod_basis.shape[1]]   = pod_basis.T.dot(matrix_no_bc[:n_dofs["space"],:n_dofs["space"]].dot(pod_basis))
#     reduced_system_matrix[pod_basis.shape[1]:,:pod_basis.shape[1]]   = pod_basis.T.dot(matrix_no_bc[n_dofs["space"]:,:n_dofs["space"]].dot(pod_basis))
#     reduced_system_matrix[:pod_basis.shape[1],   pod_basis.shape[1]:] = pod_basis.T.dot(matrix_no_bc[:n_dofs["space"],n_dofs["space"]:].dot(pod_basis))
#     reduced_system_matrix[pod_basis.shape[1]:, pod_basis.shape[1]:] = pod_basis.T.dot(matrix_no_bc[n_dofs["space"]:,n_dofs["space"]:].dot(pod_basis))
    
#     # start_time2 = time.time()
#     # reduced_jump_matrix = space_time_pod_basis.T.dot(
#     #     jump_matrix_no_bc.dot(space_time_pod_basis))
#     reduced_jump_matrix = np.zeros([pod_basis.shape[1]*2,pod_basis.shape[1]*2])
#     start_time2_2 = time.time()
#     reduced_jump_matrix[:pod_basis.shape[1],   :pod_basis.shape[1]]   = pod_basis.T.dot(jump_matrix_no_bc[:n_dofs["space"],:n_dofs["space"]].dot(pod_basis))
#     reduced_jump_matrix[pod_basis.shape[1]:,:pod_basis.shape[1]]   = pod_basis.T.dot(jump_matrix_no_bc[n_dofs["space"]:,:n_dofs["space"]].dot(pod_basis))
#     reduced_jump_matrix[:pod_basis.shape[1],   pod_basis.shape[1]:] = pod_basis.T.dot(jump_matrix_no_bc[:n_dofs["space"],n_dofs["space"]:].dot(pod_basis))
#     reduced_jump_matrix[pod_basis.shape[1]:, pod_basis.shape[1]:] = pod_basis.T.dot(jump_matrix_no_bc[n_dofs["space"]:,n_dofs["space"]:].dot(pod_basis))
 
    
    
#     # print(f"ST-POD Basis: {start_time1-start_time}")
#     # # print(f"Red SM:       {start_time1_1-start_time1}")
#     # print(f"Red SM - bw:  {start_time2-start_time1_1}")
#     # # print(f"Red JM:       {start_time2_2-start_time2}")
#     # print(f"Red JM -bw:   {time.time()-start_time2_2}")
#     extime_matrix = time.time() - start_time
    
#     # print("fom:   " + str(extime_solve_FOM/(extime_solve_FOM+extime_iPOD+extime_matrix))+ ": " + str(extime_solve_FOM))
#     # print("iPOD:  " + str(extime_iPOD/(extime_solve_FOM+extime_iPOD+extime_matrix))+ ": " + str(extime_iPOD))
#     # print("mat:   " + str(extime_matrix/(extime_solve_FOM+extime_iPOD+extime_matrix))+ ": " + str(extime_matrix))
#     return pod_basis, space_time_pod_basis, reduced_system_matrix, reduced_jump_matrix, projected_reduced_solution, singular_values_tmp, total_energy_tmp


# %% initilaize ROM framework
total_energy = 0
pod_basis = np.empty([0, 0])
bunch = np.empty([0, 0])
singular_values = np.empty([0, 0])

for slab_step in range(
        # 1):
        # onyl use first solution of slab since we assume that solutions are quite similar
        int(primal_solutions[0].shape[0] / n_dofs["space"])):
    pod_basis, bunch, singular_values, total_energy = iPOD(pod_basis, bunch, singular_values, primal_solutions[0][range(
        slab_step * n_dofs["space"], (slab_step + 1) * n_dofs["space"])], total_energy)

# change from the FOM to the POD basis
space_time_pod_basis = scipy.sparse.block_diag(
    [pod_basis] * time_dofs_per_time_interval)
dual_space_time_pod_basis = scipy.sparse.block_diag(
    [dual_pod_basis] * time_dofs_per_time_interval)
# if PLOTTING:
#plt.spy(space_time_pod_basis, markersize=2)
# plt.show()

reduced_system_matrix = space_time_pod_basis.T.dot(
    matrix_no_bc.dot(space_time_pod_basis))
reduced_jump_matrix = space_time_pod_basis.T.dot(
    jump_matrix_no_bc.dot(space_time_pod_basis))


# %% primal ROM solve
reduced_solutions = []
reduced_solutions_old = space_time_pod_basis.T.dot(primal_solutions[0])
projected_reduced_solutions = []
temporal_interval_error = []
temporal_interval_error_incidactor = []
tol = 5e-4/(n_slabs)
tol_rel = 5e-2
print("tol = " + str(tol))

start_execution = time.time()
extime_solve = 0.0
extime_error = 0.0
extime_update  =0.0
for i in range(n_slabs):
    start_time = time.time()
    reduced_rhs = space_time_pod_basis.T.dot(rhs_no_bc[i])
    if i > 0:
        reduced_rhs -= reduced_jump_matrix.dot(reduced_solutions_old)
    reduced_solutions = scipy.sparse.linalg.spsolve(
        reduced_system_matrix, reduced_rhs)
    projected_reduced_solutions.append(
        space_time_pod_basis.dot(reduced_solutions))
    extime_solve += time.time() - start_time
    
    start_time = time.time()
    # temporal localization of ROM error (using FOM dual solution)
    tmp = -matrix_no_bc.dot(projected_reduced_solutions[i]) + rhs_no_bc[i]
    if i > 0:
        tmp -= jump_matrix_no_bc.dot(projected_reduced_solutions[i - 1])
    temporal_interval_error.append(np.dot(dual_solutions[i], tmp))
    temporal_interval_error_incidactor.append(0)
    # or  np.abs(temporal_interval_error[-1]/temporal_interval_error[i-1]):
    extime_error += time.time() - start_time
        
    start_time = time.time()
    # if np.abs(temporal_interval_error[-1]) > tol:
    if np.abs(temporal_interval_error[-1])/np.abs(np.dot(projected_reduced_solutions[i], dual_rhs_no_bc[i])+temporal_interval_error[-1]) > tol_rel:
        temporal_interval_error_incidactor[-1] = 1
        pod_basis, space_time_pod_basis, reduced_system_matrix, reduced_jump_matrix, projected_reduced_solutions[-1], singular_values, total_energy = ROM_update(
            pod_basis, space_time_pod_basis, reduced_system_matrix, reduced_jump_matrix, projected_reduced_solutions[i - 1])
        #reduced_solutions[-1] = space_time_pod_basis.T.dot(projected_reduced_solutions[-1])
        reduced_solutions = space_time_pod_basis.T.dot(
            projected_reduced_solutions[-1])
        tmp = - \temporal_interval_error[-1])/np.abs(np.dot(projected_reduced_solutions[-1],  mass_matrix_no_bc.dot(projected_reduced_solutions[-1]))+temporal_interval_error[-1]
            matrix_no_bc.dot(projected_reduced_solutions[i]) + rhs_no_bc[i]
        if i > 0:
            tmp -= jump_matrix_no_bc.dot(
                projected_reduced_solutions[i - 1])
        temporal_interval_error[-1] = np.dot(dual_solutions[i], tmp)
        if np.abs(temporal_interval_error[-1]) > tol:
            print('Error correction failed')
    reduced_solutions_old = reduced_solutions
    extime_update += time.time() - start_time
end_execution = time.time()
execution_time_ROM = end_execution - start_execution
print("FOM time:   " + str(execution_time_FOM))
print("ROM time:   " + str(execution_time_ROM))
print("speedup:    " + str(execution_time_FOM/execution_time_ROM))
print("Size ROM:   " + str(pod_basis.shape[1]))
print("FOM solves: " + str(np.sum(temporal_interval_error_incidactor)
                           ) + " / " + str(len(temporal_interval_error_incidactor)))

print("ROM Solve time: " + str(extime_solve))
print("Error est time: " + str(extime_error))
print("Update time:    " + str(extime_update))
print(str(extime_solve+extime_error+extime_update))

projected_reduced_solution = np.hstack(projected_reduced_solutions)

original_stdout = sys.stdout  # Save a reference to the original standard output
# with open('output/speedup_' + CASE + '_cycle_' + str(cycle) + '.txt', 'a') as f:
#     sys.stdout = f # Change the standard output to the file we created.
#     print(str(execution_time_FOM) + ', ' + str(execution_time_ROM) + ', ' + str(execution_time_FOM/execution_time_ROM))
#     sys.stdout = original_stdout # Reset the standard output to its original value


J_r = 0.
J_r_t = np.empty([n_slabs, 1])
for i in range(n_slabs):
    J_r_t[i] = np.dot(projected_reduced_solutions[i], dual_rhs_no_bc[i])
    J_r += np.dot(projected_reduced_solutions[i], dual_rhs_no_bc[i])
J["u_r"].append(J_r)

print("J(u_h) =", J["u_h"])
# TODO: in the future compare J(u_r) for different values of r
print("J(u_r) =", J["u_r"][-1])

# %% Ploting
# Plot 3: temporal error
# WARNING: hardcoding end time T = 4.
time_step_size = 4.0 / (n_dofs["time"] / 2)
xx, yy = [], []
xx_FOM, yy_FOM = [], []
cc = []
for i, error in enumerate(temporal_interval_error):
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
plt.ylabel("$\eta\\raisebox{-.5ex}{$|$}_{Q_l}$")
plt.yscale("log")
plt.xlim([0, n_slabs*time_step_size])
#plt.title("temporal evaluation of cost funtional")
plt.savefig(SAVE_PATH + "temporal_error_cost_funtional.eps", format='eps')
plt.savefig(SAVE_PATH + "temporal_error_cost_funtional.png", format='png')

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

reduced_dual_matrix = dual_space_time_pod_basis.T.dot(matrix_no_bc.T.dot(dual_space_time_pod_basis))
reduced_dual_jump_matrix = dual_space_time_pod_basis.T.dot(
    jump_matrix_no_bc.T.dot(dual_space_time_pod_basis))
# --------------
# %% dual ROM solve
reduced_dual_solutions = []
projected_reduced_dual_solutions = []
for i in list(range(n_slabs))[::-1]:
    reduced_dual_rhs = dual_space_time_pod_basis.T.dot(dual_rhs_no_bc[i])
    if len(reduced_dual_solutions):
        reduced_dual_rhs -= reduced_dual_jump_matrix.dot(
            reduced_dual_solutions[-1])
    reduced_dual_solutions.append(scipy.sparse.linalg.spsolve(
        reduced_dual_matrix, reduced_dual_rhs))
    projected_reduced_dual_solutions.append(
        dual_space_time_pod_basis.dot(reduced_dual_solutions[-1]))

projected_reduced_dual_solutions = projected_reduced_dual_solutions[::-1]
projected_reduced_dual_solution = np.hstack(
    projected_reduced_dual_solutions)
# ----------------
# %% error estimation
true_error = J['u_h'] - J['u_r'][-1]

# using FOM dual solution
print("\nUsing z_h:")
print("----------")
error_estimator = sum(temporal_interval_error)
print(f"True error:        {true_error}")
print(f"Estimated error:   {error_estimator}")
print(f"Effectivity index: {abs(true_error / error_estimator)}")
# using ROM dual solution
temporal_interval_error_cheap = []
for i in range(n_slabs):
    # temporal localization of ROM error (using ROM dual solution)
    tmp = -matrix_no_bc.dot(projected_reduced_solutions[i]) + rhs_no_bc[i]
    if i >= 1:
        tmp -= jump_matrix_no_bc.dot(projected_reduced_solutions[i - 1])
    temporal_interval_error_cheap.append(
        np.dot(projected_reduced_dual_solutions[i], tmp))

print("\nUsing z_r:")
print("----------")
error_estimator_cheap = sum(temporal_interval_error_cheap)
print(f"True error:              {true_error}")
print(f"Cheap estimated error:   {error_estimator_cheap}")
print(
    f"Effectivity index:       {abs(true_error / error_estimator_cheap)}")
