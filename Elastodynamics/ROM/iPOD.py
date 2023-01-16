import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.linalg
import time
import multiprocessing

def iPOD(POD, bunch, singular_values, snapshot, total_energy,energy_content,bunch_size):
    # bunch_size = 2
    # energy_content = 0.9999 #1 -1e-8 # 0.9999
    row, col_POD = POD.shape

    if (bunch.shape[1] == 0):
        bunch = np.empty([np.shape(snapshot)[0], 0])

    bunch = np.hstack((bunch, snapshot.reshape(-1, 1)))
    _, col = bunch.shape

    total_energy += np.dot((snapshot), (snapshot))
    if (col == bunch_size):
        if (col_POD == 0):
            POD, S, _ = scipy.linalg.svd(bunch, full_matrices=False)
            r = 0
            # np.shape(S_k)[0])):
            while ((np.dot(S[0:r], S[0:r]) / total_energy <= energy_content) and (r <= bunch_size)):
                r += 1
                # print(r)
            singular_values = S[0:r]
            POD = POD[:, 0:r]
        else:
            M = np.dot(POD.T, bunch)
            P = bunch - np.dot(POD, M)

            Q_p, R_p = scipy.linalg.qr(P, mode='economic')
            Q_q = np.hstack((POD, Q_p))

            S0 = np.vstack((np.diag(singular_values), np.zeros(
                (np.shape(R_p)[0], np.shape(singular_values)[0]))))
            MR_p = np.vstack((M, R_p))
            K = np.hstack((S0, MR_p))

            # (np.linalg.norm(np.matmul(Q_q.T, Q_q) -np.eye(np.shape(Q_q)[1])) >= 1e-14):
            # if (True):
            if (np.inner(Q_q[:,0],Q_q[:,-1]) >= 1e-10):
                Q_q, R_q = scipy.linalg.qr(Q_q, mode='economic')
                K = np.matmul(R_q, K)

            U_k, S_k, _ = scipy.linalg.svd(K, full_matrices=False)
            # Q = np.hstack((POD, Q_p))
            # Q_q, _ = scipy.linalg.qr(Q, mode='economic')

            r = 0# col_POD + 1 #0
            while ((np.dot(S_k[0:r], S_k[0:r]) / total_energy <=
                   energy_content) and (r < np.shape(S_k)[0])):
                r += 1
                # print(r)

            singular_values = S_k[0:r]
            POD = np.matmul(Q_q, U_k[:, 0:r])
        bunch = np.empty([np.shape(bunch)[0], 0])
        # print(np.shape(POD))

    return POD, bunch, singular_values, total_energy


def ROM_update(
        pod_basis,
        # space_time_pod_basis,
        reduced_system_matrix,
        last_projected_reduced_solution,
        primal_rhs, # with bc enforced
        primal_matrix, # with bc enforced
        singular_values,
        total_energy,
        n_dofs,
        time_dofs_per_time_interval,
        matrix_no_bc,
        energy_content):
    # creating primal rhs and applying BC to it
    
    extime_solve_FOM = 0.0
    extime_iPOD = 0.0
    extime_matrix = 0.0
    
    start_time = time.time()
    
    ## Solve FOM step
    primal_rhs = primal_system_rhs[i][n_dofs["time_step"]:].copy() - C_wbc.dot(last_primal_solution)
    primal_solution = scipy.sparse.linalg.spsolve(D_wbc, primal_rhs)
    
    for j in (ordering_on_slab[1:]-1): #[l-1 for l in ordering_on_slab[1:]]: #range(n_dofs["solperstep"]):
        primal_solutions.append(primal_solution[j*n_dofs["time_step"]:(j+1)*n_dofs["time_step"]])
        
    last_primal_solution = primal_solutions[-1]
    
    
    primal_rhs -= jump_matrix_no_bc.dot(last_projected_reduced_solution)
    for row in boundary_ids:
        primal_rhs[row] = 0.  # NOTE: hardcoding homogeneous Dirichlet BC

    projected_reduced_solution = scipy.sparse.linalg.spsolve(
        primal_matrix, primal_rhs)
    extime_solve_FOM = time.time() - start_time
    
    start_time = time.time()
    singular_values_tmp = singular_values
    total_energy_tmp = total_energy
    bunch_tmp = np.empty([0, 0])
    if ((
            projected_reduced_solution.shape[0] / n_dofs["space"]).is_integer()):
        # onyl use first solution of slab since we assume that solutions are quite similar
        for slab_step in range(
                # 1):
                int(projected_reduced_solution.shape[0] / n_dofs["space"])):
            pod_basis, bunch_tmp, singular_values_tmp, total_energy_tmp = iPOD(pod_basis, bunch_tmp, singular_values_tmp, projected_reduced_solution[range(
                slab_step * n_dofs["space"], (slab_step + 1) * n_dofs["space"])], total_energy_tmp,energy_content)
    else:
        print(
            (projected_reduced_solution.shape[0] / n_dofs["space"]).is_integer())
        print("Error building slapwise POD")
    extime_iPOD = time.time() - start_time
    
    start_time = time.time()
    # change from the FOM to the POD basis
    # space_time_pod_basis = scipy.sparse.block_diag(
    #     [pod_basis] * time_dofs_per_time_interval)
    
    # start_time1 = time.time()
    # reduced_system_matrix = space_time_pod_basis.T.dot(
    #     matrix_no_bc.dot(space_time_pod_basis))
    
    reduced_system_matrix = reduce_matrix(matrix_no_bc,pod_basis,pod_basis)
    reduced_jump_matrix = reduce_matrix(jump_matrix_no_bc,pod_basis,pod_basis)
    
    extime_matrix = time.time() - start_time
    
    # print("fom - prim:  " + str(extime_solve_FOM/(extime_solve_FOM+extime_iPOD+extime_matrix))+ ": " + str(extime_solve_FOM))
    # print("iPOD - prim: " + str(extime_iPOD/(extime_solve_FOM+extime_iPOD+extime_matrix))+ ": " + str(extime_iPOD))
    # print("mat - prim:  " + str(extime_matrix/(extime_solve_FOM+extime_iPOD+extime_matrix))+ ": " + str(extime_matrix))
    # return pod_basis, space_time_pod_basis, reduced_system_matrix, reduced_jump_matrix, projected_reduced_solution, singular_values_tmp, total_energy_tmp
    return pod_basis, reduced_system_matrix, reduced_jump_matrix, projected_reduced_solution, singular_values_tmp, total_energy_tmp

def ROM_update_dual(
        pod_basis,
        # space_time_pod_basis,
        reduced_system_matrix,
        reduced_jump_matrix,
        last_projected_reduced_solution,
        dual_rhs,
        jump_matrix_no_bc,
        boundary_ids,
        primal_matrix,
        singular_values,
        total_energy,
        n_dofs,
        time_dofs_per_time_interval,
        matrix_no_bc,
        energy_content):
    # creating primal rhs and applying BC to it
    
    extime_solve_FOM = 0.0
    extime_iPOD = 0.0
    extime_matrix = 0.0
    
    start_time = time.time()
    dual_rhs -= jump_matrix_no_bc.T.dot(last_projected_reduced_solution)
                  # jump_matrix_no_bc.T.dot(space_time_pod_basis_dual.dot(forwarded_reduced_dual_solutions[-2]))
    for row in boundary_ids:
        dual_rhs[row] = 0.  # NOTE: hardcoding homogeneous Dirichlet BC

    projected_reduced_solution = scipy.sparse.linalg.spsolve(
        primal_matrix, dual_rhs)
    extime_solve_FOM = time.time() - start_time
    
    start_time = time.time()
    singular_values_tmp = singular_values
    total_energy_tmp = total_energy
    bunch_tmp = np.empty([0, 0])
    if ((
            projected_reduced_solution.shape[0] / n_dofs["space"]).is_integer()):
        # onyl use first solution of slab since we assume that solutions are quite similar
        for slab_step in range(
                # 1):
                int(projected_reduced_solution.shape[0] / n_dofs["space"])):
            pod_basis, bunch_tmp, singular_values_tmp, total_energy_tmp = iPOD(pod_basis, bunch_tmp, singular_values_tmp, projected_reduced_solution[range(
                slab_step * n_dofs["space"], (slab_step + 1) * n_dofs["space"])], total_energy_tmp,energy_content)
    else:
        print(
            (projected_reduced_solution.shape[0] / n_dofs["space"]).is_integer())
        print("Error building slapwise POD")
    extime_iPOD = time.time() - start_time
    
    start_time = time.time()
    # change from the FOM to the POD basis
    # space_time_pod_basis = scipy.sparse.block_diag(
    #     [pod_basis] * time_dofs_per_time_interval)
    
    reduced_system_matrix = reduce_matrix(matrix_no_bc,pod_basis,pod_basis)
    reduced_jump_matrix = reduce_matrix(jump_matrix_no_bc,pod_basis,pod_basis)
    
    
    # pool = multiprocessing.Pool(4)
    # processes = []
    # processes.append(pool.apply_async(reduce_matrix, args=(matrix_no_bc,pod_basis,pod_basis,)))
    # processes.append(pool.apply_async(reduce_matrix, args=(jump_matrix_no_bc,pod_basis,pod_basis,)))

    # result = [p.get() for p in processes]
    
    # reduced_system_matrix   = result[0] #   pod_basis_left.T.dot(matrix[:n_h_dofs,:n_h_dofs].dot(pod_basis_right))
    # reduced_jump_matrix  = result[1]  #pod_basis_left.T.dot(matrix[n_h_dofs:,:n_h_dofs].dot(pod_basis_right))

    # start_time1 = time.time()
    # reduced_system_matrix = space_time_pod_basis.T.dot(
    #     matrix_no_bc.dot(space_time_pod_basis))
    # reduced_system_matrix = np.zeros([pod_basis.shape[1]*2,pod_basis.shape[1]*2])
    # start_time1_1 = time.time()
    # reduced_system_matrix[:pod_basis.shape[1],   :pod_basis.shape[1]]   = pod_basis.T.dot(matrix_no_bc[:n_dofs["space"],:n_dofs["space"]].dot(pod_basis))
    # reduced_system_matrix[pod_basis.shape[1]:,:pod_basis.shape[1]]   = pod_basis.T.dot(matrix_no_bc[n_dofs["space"]:,:n_dofs["space"]].dot(pod_basis))
    # reduced_system_matrix[:pod_basis.shape[1],   pod_basis.shape[1]:] = pod_basis.T.dot(matrix_no_bc[:n_dofs["space"],n_dofs["space"]:].dot(pod_basis))
    # reduced_system_matrix[pod_basis.shape[1]:, pod_basis.shape[1]:] = pod_basis.T.dot(matrix_no_bc[n_dofs["space"]:,n_dofs["space"]:].dot(pod_basis))
    
    # # start_time2 = time.time()
    # # reduced_jump_matrix = space_time_pod_basis.T.dot(
    # #     jump_matrix_no_bc.dot(space_time_pod_basis))
    # reduced_jump_matrix = np.zeros([pod_basis.shape[1]*2,pod_basis.shape[1]*2])
    # start_time2_2 = time.time()
    # reduced_jump_matrix[:pod_basis.shape[1],   :pod_basis.shape[1]]   = pod_basis.T.dot(jump_matrix_no_bc[:n_dofs["space"],:n_dofs["space"]].dot(pod_basis))
    # reduced_jump_matrix[pod_basis.shape[1]:,:pod_basis.shape[1]]   = pod_basis.T.dot(jump_matrix_no_bc[n_dofs["space"]:,:n_dofs["space"]].dot(pod_basis))
    # reduced_jump_matrix[:pod_basis.shape[1],   pod_basis.shape[1]:] = pod_basis.T.dot(jump_matrix_no_bc[:n_dofs["space"],n_dofs["space"]:].dot(pod_basis))
    # reduced_jump_matrix[pod_basis.shape[1]:, pod_basis.shape[1]:] = pod_basis.T.dot(jump_matrix_no_bc[n_dofs["space"]:,n_dofs["space"]:].dot(pod_basis))
 
    
    
    # print(f"ST-POD Basis: {start_time1-start_time}")
    # # print(f"Red SM:       {start_time1_1-start_time1}")
    # print(f"Red SM - bw:  {start_time2-start_time1_1}")
    # # print(f"Red JM:       {start_time2_2-start_time2}")
    # print(f"Red JM -bw:   {time.time()-start_time2_2}")
    extime_matrix = time.time() - start_time
    
    # print("fom - dual:   " + str(extime_solve_FOM/(extime_solve_FOM+extime_iPOD+extime_matrix))+ ": " + str(extime_solve_FOM))
    # print("iPOD - dual:  " + str(extime_iPOD/(extime_solve_FOM+extime_iPOD+extime_matrix))+ ": " + str(extime_iPOD))
    # print("mat- dual:    " + str(extime_matrix/(extime_solve_FOM+extime_iPOD+extime_matrix))+ ": " + str(extime_matrix))
    # print(" ")
    # return pod_basis, space_time_pod_basis, reduced_system_matrix, reduced_jump_matrix, projected_reduced_solution, singular_values_tmp, total_energy_tmp
    return pod_basis, reduced_system_matrix, reduced_jump_matrix, projected_reduced_solution, singular_values_tmp, total_energy_tmp


def reduction(pod_basis_left,matrix,pod_basis_right):
    return pod_basis_left.T.dot(matrix.dot(pod_basis_right))


def reduce_sub_matrix(matrix, pod_basis_left, pod_basis_right):
    
    n_dofs = pod_basis_left["displacement"].shape[0]
    size_u = pod_basis_left["displacement"].shape[1]
    size_v = pod_basis_left["velocity"].shape[1]
    size_u_right = pod_basis_right["displacement"].shape[1]
    size_v_right = pod_basis_right["velocity"].shape[1]
    
    reduced_matrix = np.zeros([size_u + size_v, size_u_right + size_v_right])
    
    reduced_matrix[:size_u,:size_u_right] = pod_basis_left["displacement"].T.dot(matrix[:n_dofs,:n_dofs].dot(pod_basis_right["displacement"]))
    reduced_matrix[size_u:,size_u_right:] = pod_basis_left["velocity"].T.dot(matrix[n_dofs:,n_dofs:].dot(pod_basis_right["velocity"]))
    reduced_matrix[:size_u,size_u_right:] = pod_basis_left["displacement"].T.dot(matrix[:n_dofs,n_dofs:].dot(pod_basis_right["velocity"]))
    reduced_matrix[size_u:,:size_u_right] = pod_basis_left["velocity"].T.dot(matrix[n_dofs:,:n_dofs].dot(pod_basis_right["displacement"]))
    
    return reduced_matrix
    
def reduce_matrix(matrix, pod_basis_left, pod_basis_right):
    
    n_dofs = pod_basis_left["displacement"].shape[0]
    
    size_u = pod_basis_left["displacement"].shape[1]
    size_v = pod_basis_left["velocity"].shape[1]
    size_u_right = pod_basis_right["displacement"].shape[1]
    size_v_right = pod_basis_right["velocity"].shape[1]
    
    mat_per_row = int(matrix.shape[0]/(2*n_dofs))
    mat_per_col = int(matrix.shape[1]/(2*n_dofs))


    n_rows_sub_mat = 2*n_dofs
    n_cols_sub_mat = 2*n_dofs

    n_rows_red_sub_mat = size_u + size_v
    n_cols_red_sub_mat = size_u_right + size_v_right

    reduced_matrix = np.zeros([mat_per_row*n_rows_red_sub_mat, mat_per_col*n_cols_red_sub_mat])
    
    for i in range(mat_per_row):
        for j in range(mat_per_col):
            # print(f"row - rom: {i*n_rows_red_sub_mat}, {(i+1)*n_rows_red_sub_mat} ")
            # print(f"col - rom: {j*n_cols_red_sub_mat}, {(j+1)*n_cols_red_sub_mat} ")
            
            # print(f"row - fom: {i*n_rows_sub_mat}, {(i+1)*n_rows_sub_mat} ")
            # print(f"col - fom: {j*n_cols_sub_mat}, {(j+1)*n_cols_sub_mat} ")
            reduced_matrix[i*n_rows_red_sub_mat:(i+1)*n_rows_red_sub_mat, \
                           j*n_cols_red_sub_mat:(j+1)*n_cols_red_sub_mat] \
                = reduce_sub_matrix(matrix[i*n_rows_sub_mat:(i+1)*n_rows_sub_mat, \
                                           j*n_cols_sub_mat:(j+1)*n_cols_sub_mat], \
                                    pod_basis_left,pod_basis_right)

    # print("cg:")
    # print(np.max([n_sub_u, n_sub_v]))
    
    # n_dofs = pod_basis_left["displacement"].shape[0]
    # size_u = pod_basis_left["displacement"].shape[1]
    # size_v = pod_basis_left["velocity"].shape[1]
    # size_u_right = pod_basis_right["displacement"].shape[1]
    # size_v_right = pod_basis_right["velocity"].shape[1]
    
    # reduced_matrix = np.zeros([size_u + size_v, size_u_right + size_v_right])
    
    # reduced_matrix[:size_u,:size_u_right] = pod_basis_left["displacement"].T.dot(matrix[:n_dofs,:n_dofs].dot(pod_basis_right["displacement"]))
    # reduced_matrix[size_u:,size_u_right:] = pod_basis_left["velocity"].T.dot(matrix[n_dofs:,n_dofs:].dot(pod_basis_right["velocity"]))
    # reduced_matrix[:size_u,size_u_right:] = pod_basis_left["displacement"].T.dot(matrix[:n_dofs,n_dofs:].dot(pod_basis_right["velocity"]))
    # reduced_matrix[size_u:,:size_u_right] = pod_basis_left["velocity"].T.dot(matrix[n_dofs:,:n_dofs].dot(pod_basis_right["displacement"]))
    
    return reduced_matrix


def reduce_sub_vector(vector, pod_basis):
    
    n_dofs = pod_basis["displacement"].shape[0]
    size_u = pod_basis["displacement"].shape[1]
    size_v = pod_basis["velocity"].shape[1]
    
    reduced_vector = np.zeros([size_u + size_v, ])
    
    reduced_vector[:size_u] = pod_basis["displacement"].T.dot(vector[:n_dofs])
    reduced_vector[size_u:] = pod_basis["velocity"].T.dot(vector[n_dofs:])
    
    return reduced_vector
    
def reduce_vector(vector, pod_basis):
    
    n_dofs = pod_basis["displacement"].shape[0]
    size_u = pod_basis["displacement"].shape[1]
    size_v = pod_basis["velocity"].shape[1]
    
    n_rows_sub_vec = 2*n_dofs
    vec_per_row = int(vector.shape[0]/(n_rows_sub_vec))
    n_rows_sub_red_vec = size_u + size_v
    
    reduced_vector = np.zeros([vec_per_row*n_rows_sub_red_vec, ])
    
    for i in range(vec_per_row):
        reduced_vector[i*n_rows_sub_red_vec:(i+1)*n_rows_sub_red_vec] = \
            reduce_sub_vector(vector[i*n_rows_sub_vec:(i+1)*n_rows_sub_vec],pod_basis)
    
    return reduced_vector
    
def project_vector(reduced_vector, pod_basis): 
          
    n_dofs_u = pod_basis["displacement"].shape[0]
    n_dofs_v = pod_basis["velocity"].shape[0]
    
    size_u = pod_basis["displacement"].shape[1]
    size_v = pod_basis["velocity"].shape[1]
    
    vector = np.zeros([n_dofs_u + n_dofs_v,])

    vector[:n_dofs_u] = pod_basis["displacement"].dot(reduced_vector[:size_u])
    vector[n_dofs_u:] = pod_basis["velocity"].dot(reduced_vector[size_u:])
    
    return vector    
    
