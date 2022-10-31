import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.linalg
import scipy.interpolate
import matplotlib.pyplot as plt
import os
import time


PLOTTING = True
INTERPOLATION_TYPE = "nearest"  # "linear", "cubic"
LOAD_PRIMAL_SOLUTION = False
OUTPUT_PATH = "../../FOM/slabwise/output/dim=1/"


def iPOD(POD, bunch, singular_values, snapshot, total_energy):
    bunch_size = 2
    energy_content = 1 -1e-8 # 0.9999
    row, col_POD = POD.shape

    if (bunch.shape[1] == 0):
        bunch = np.empty([np.shape(snapshot)[0], 0])

    bunch = np.hstack((bunch, snapshot.reshape(-1, 1)))
    _, col = bunch.shape

    total_energy += np.dot(snapshot, snapshot)
    if (col == bunch_size):
        if (col_POD == 0):
            POD, S, _ = scipy.linalg.svd(bunch, full_matrices=False)
            r = 0
            # np.shape(S_k)[0])):
            while ((np.dot(S[0:r], S[0:r]) / total_energy <=
                   energy_content) and (r <= bunch_size)):
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
            if (True):
                Q_q, R_q = scipy.linalg.qr(Q_q, mode='economic')
                K = np.matmul(R_q, K)

            U_k, S_k, _ = scipy.linalg.svd(K, full_matrices=False)
            Q = np.hstack((POD, Q_p))
            Q_q, _ = scipy.linalg.qr(Q, mode='economic')

            r = col_POD + 1 #0
            while ((np.dot(S_k[0:r], S_k[0:r]) / total_energy <=
                   energy_content) and (r < np.shape(S_k)[0])):
                r += 1
                # print(r)

            singular_values = S_k[0:r]
            POD = np.matmul(Q_q, U_k[:, 0:r])
        bunch = np.empty([np.shape(bunch)[0], 0])
        # print(np.shape(POD))

    return POD, bunch, singular_values, total_energy


for cycle in os.listdir(OUTPUT_PATH):
    if "7" not in cycle:
        continue
    print(f"\n{'-'*12}\n| {cycle}: |\n{'-'*12}\n")
    # NO BC
    [data, row, column] = np.loadtxt(OUTPUT_PATH + cycle + "/matrix_no_bc.txt")
    matrix_no_bc = scipy.sparse.csr_matrix(
        (data, (row.astype(int), column.astype(int))))

    [data, row, column] = np.loadtxt(
        OUTPUT_PATH + cycle + "/jump_matrix_no_bc.txt")
    jump_matrix_no_bc = scipy.sparse.csr_matrix(
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
    n_dofs = {"space": coordinates_x.shape[0], "time": coordinates_t.shape[0]}

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
        # if ((primal_solutions[-1].shape[0] / n_dofs["space"]).is_integer()):
        #     for slab_step in range(
        #             int(primal_solutions[-1].shape[0] / n_dofs["space"])):
        #         # POD, bunch, singular_values, total_energy = iPOD(
        #         # POD, bunch, singular_values,
        #         # primal_solutions[-1][range(n_dofs["space"])], total_energy)
        #         POD, bunch, singular_values, total_energy = iPOD(POD, bunch, singular_values, primal_solutions[-1][range(
        #             slab_step * n_dofs["space"], (slab_step + 1) * n_dofs["space"])], total_energy)
        # else:
        #     print("Error building slapwise POD")
        last_primal_solution = primal_solutions[-1]
    end_execution = time.time()
    execution_time_FOM = end_execution - start_execution
    # plot primal solution
    primal_solution = np.hstack(primal_solutions)
    if PLOTTING:
        grid_t, grid_x = np.mgrid[0:4:100j, 0:1:100j]
        primal_grid = scipy.interpolate.griddata(
            coordinates, primal_solution, (grid_t, grid_x), method=INTERPOLATION_TYPE)
        plt.title(f"Primal solution (ref={cycle.split('=')[1]})")
        plt.imshow(primal_grid.T, extent=(0, 4, 0, 1), origin='lower')
        plt.xlabel("$t$")
        plt.ylabel("$x$")
        plt.colorbar()
        plt.show()

    # ----------
    # %% dual solve
    last_dual_solution = np.zeros_like(dual_rhs_no_bc[0])
    dual_solutions = []
    for i in list(range(n_slabs))[::-1]:
        # creating dual rhs and applying BC to it
        dual_rhs = dual_rhs_no_bc[i].copy()
        dual_rhs -= jump_matrix_no_bc.T.dot(last_dual_solution)
        for row in boundary_ids:
            dual_rhs[row] = 0.

        dual_solutions.append(
            scipy.sparse.linalg.spsolve(dual_matrix, dual_rhs))

        last_dual_solution = dual_solutions[-1]

    # dual solution
    dual_solutions = dual_solutions[::-1]
    dual_solution = np.hstack(dual_solutions)
    if PLOTTING:
        grid_t, grid_x = np.mgrid[0:4:100j, 0:1:100j]
        dual_grid = scipy.interpolate.griddata(
            coordinates, dual_solution, (grid_t, grid_x), method=INTERPOLATION_TYPE)
        plt.title(f"Dual solution (ref={cycle.split('=')[1]})")
        plt.imshow(dual_grid.T, extent=(0, 4, 0, 1), origin='lower')
        plt.xlabel("$t$")
        plt.ylabel("$x$")
        plt.colorbar()
        plt.show()

    # %% goal functionals
    J = {"u_h": 0., "u_r": []}
    J_h_t = np.empty([n_slabs,1])
    for i in range(n_slabs):
        J_h_t[i] = np.dot(primal_solutions[i], dual_rhs_no_bc[i])
        J["u_h"] += np.dot(primal_solutions[i], dual_rhs_no_bc[i])

    # ---------------- #
    #%% PRIMAL POD-ROM #
    # ---------------- #
    print(f"#DoFs(space): {n_dofs['space']}")
    print(f"#DoFs(time): {n_dofs['time']}")

    # ---------------
    # snapshot matrix
    Y = primal_solution.reshape(n_dofs["time"], n_dofs["space"]).T
    Y_dual = dual_solution.reshape(n_dofs["time"], n_dofs["space"]).T

    # ------------------
    # correlation matrix
    # TODO: include mass matrix
    correlation_matrix = np.dot(Y.T, Y)
    dual_correlation_matrix = np.dot(Y_dual.T, Y_dual)
    assert correlation_matrix.shape == (
        n_dofs["time"], n_dofs["time"]), f"Matrix should be of #DoFs(time) shape"

    # -----------------------------------------
    # compute eigenvalues of correlation matrix
    eigen_values, eigen_vectors = np.linalg.eig(correlation_matrix)
    dual_eigen_values, dual_eigen_vectors = np.linalg.eig(
        dual_correlation_matrix)

    # ignore the complex part of the eigen values and eigen vectors
    eigen_values, eigen_vectors = eigen_values.astype(
        'float64'), eigen_vectors.astype('float64')
    dual_eigen_values, dual_eigen_vectors = dual_eigen_values.astype(
        'float64'), dual_eigen_vectors.astype('float64')

    # sort the eigenvalues in descending order
    eigen_values, eigen_vectors = eigen_values[eigen_values.argsort(
    )[::-1]], eigen_vectors[eigen_values.argsort()[::-1]]
    dual_eigen_values, dual_eigen_vectors = dual_eigen_values[dual_eigen_values.argsort(
    )[::-1]], dual_eigen_vectors[eigen_values.argsort()[::-1]]

    POD_full, singular_values_test, _ = scipy.linalg.svd(
        Y, full_matrices=False)
    POD_full = POD_full[:, 0:np.shape(singular_values)[0]]
    # plot eigen value decay
    if PLOTTING:
        plt.plot(range(1,
                       min(n_dofs["space"],
                           n_dofs["time"]) + 1),
                 eigen_values,
                 label="correlation matrix")
        plt.plot(
            range(
                1,
                np.shape(singular_values)[0] +
                1),
            np.power(
                singular_values,
                2),
            label="iSVD")
        plt.plot(range(1,
                       np.shape(singular_values)[0] + 1),
                 np.power(singular_values_test[0:np.shape(singular_values)[0]],
                          2),
                 label="SVD")
        plt.legend()
        plt.yscale("log")
        plt.xlabel("index")
        plt.ylabel("eigen value")
        plt.title("Eigenvalues of correlation matrix")
        plt.show()

        plt.plot(
            range(
                1, min(
                    n_dofs["space"], n_dofs["time"]) + 1), dual_eigen_values)
        plt.yscale("log")
        plt.xlabel("index")
        plt.ylabel("eigen value")
        plt.title("Dual eigenvalues of correlation matrix")
        plt.show()

    # desired energy ratio
    ENERGY_RATIO_THRESHOLD = 0.9999
    ENERGY_RATIO_THRESHOLD_DUAL = 0.99999999

    # determine number of POD basis vectors needed to preserve certain ratio
    # of energy
    r = np.sum([(eigen_values[:i].sum() / eigen_values.sum()) <
               ENERGY_RATIO_THRESHOLD for i in range(n_dofs["time"])])
    # FOR DEBUGGING: r = 2
    r_dual = np.sum([(dual_eigen_values[:i].sum() / dual_eigen_values.sum()) <
                     ENERGY_RATIO_THRESHOLD_DUAL for i in range(n_dofs["time"])])

    print(
        f"To preserve {ENERGY_RATIO_THRESHOLD} of information we need {r} primal POD vector(s). (result: {eigen_values[:r].sum() / eigen_values.sum()} information).")
    print(
        f"To preserve {ENERGY_RATIO_THRESHOLD_DUAL} of information we need {r_dual} dual POD vector(s). (result: {dual_eigen_values[:r_dual].sum() / dual_eigen_values.sum()} information).")

    # -------------------------
    # compute POD basis vectors
    pod_basis = np.dot(np.dot(Y, eigen_vectors[:, :r]), np.diag(
        1. / np.sqrt(eigen_values[:r])))

    dual_pod_basis = np.dot(np.dot(Y_dual, dual_eigen_vectors[:, :r_dual]), np.diag(
        1. / np.sqrt(dual_eigen_values[:r_dual])))
    # choose iSVD basis instead of correlation
    pod_basis = POD
    r = POD.shape[1]
    colors = ["red", "blue", "green", "purple", "orange", "pink", "black"]
    # plot the POD basis
    if PLOTTING:
        for i in range(r):
            plt.plot(coordinates_x,
                     pod_basis[:,
                               i],
                     color=colors[i % len(colors)],
                     label=str(i + 1))
        plt.legend()
        plt.xlabel("$x$")
        plt.title("Primal POD vectors")
        plt.show()

        for i in range(r_dual):
            plt.plot(coordinates_x, dual_pod_basis[:, i], color=colors[i % len(
                colors)], label=str(i + 1))
        plt.legend()
        plt.xlabel("$x$")
        plt.title("Dual POD vectors")
        plt.show()

    time_dofs_per_time_interval = int(n_dofs["time"] / n_slabs)
    dofs_per_time_interval = time_dofs_per_time_interval * n_dofs["space"]

    def ROM_update(
            pod_basis,
            space_time_pod_basis,
            reduced_system_matrix,
            reduced_jump_matrix,
            last_projected_reduced_solution):
        # creating primal rhs and applying BC to it
        primal_rhs = rhs_no_bc[i].copy()
        primal_rhs -= jump_matrix_no_bc.dot(last_projected_reduced_solution)
        for row in boundary_ids:
            primal_rhs[row] = 0.  # NOTE: hardcoding homogeneous Dirichlet BC

        projected_reduced_solution = scipy.sparse.linalg.spsolve(
            primal_matrix, primal_rhs)
        singular_values_tmp = singular_values
        total_energy_tmp = total_energy
        bunch_tmp = np.empty([0, 0])
        if ((
                projected_reduced_solution.shape[0] / n_dofs["space"]).is_integer()):
            # onyl use first solution of slab since we assume that solutions are quite similar
            for slab_step in range(
                    #1):  
                    int(projected_reduced_solution.shape[0] / n_dofs["space"])):
                pod_basis, bunch_tmp, singular_values_tmp, total_energy_tmp = iPOD(pod_basis, bunch_tmp, singular_values_tmp, projected_reduced_solution[range(
                    slab_step * n_dofs["space"], (slab_step + 1) * n_dofs["space"])], total_energy_tmp)
        else:
            print(
                (projected_reduced_solution.shape[0] / n_dofs["space"]).is_integer())
            print("Error building slapwise POD")

        # change from the FOM to the POD basis
        space_time_pod_basis = scipy.sparse.block_diag(
            [pod_basis] * time_dofs_per_time_interval)

        reduced_system_matrix = space_time_pod_basis.T.dot(
            matrix_no_bc.dot(space_time_pod_basis))

        reduced_jump_matrix = space_time_pod_basis.T.dot(
            jump_matrix_no_bc.dot(space_time_pod_basis))

        return pod_basis, space_time_pod_basis, reduced_system_matrix, reduced_jump_matrix, projected_reduced_solution, singular_values_tmp, total_energy_tmp

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

    # plot the initial POD basis
    for i in range(pod_basis.shape[1]):
        plt.plot(coordinates_x,
                 pod_basis[:,
                           i],
                 color=colors[i % len(colors)],
                 label=str(i + 1))
    plt.legend()
    plt.xlabel("$x$")
    plt.title("DEBUG: Initial POD vectors")
    plt.show()

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

    # ----------------
    # %% primal ROM solve
    reduced_solutions = []
    reduced_solutions_old = space_time_pod_basis.T.dot(primal_solutions[0])
    projected_reduced_solutions = []
    temporal_interval_error = []
    temporal_interval_error_incidactor = []
    tol =  5e-4/(n_slabs)
    tol_rel = 1e-2
    print("tol = " + str(tol))
    start_execution = time.time()
    for i in range(n_slabs):
        reduced_rhs = space_time_pod_basis.T.dot(rhs_no_bc[i])
        if i > 0:
            reduced_rhs -= reduced_jump_matrix.dot(reduced_solutions_old)
        # reduced_solutions.append(scipy.sparse.linalg.spsolve(
            # reduced_system_matrix, reduced_rhs))
        reduced_solutions = scipy.sparse.linalg.spsolve(
            reduced_system_matrix, reduced_rhs)
        # projected_reduced_solutions.append(
        #     space_time_pod_basis.dot(reduced_solutions[-1]))
        projected_reduced_solutions.append(
            space_time_pod_basis.dot(reduced_solutions))

        # temporal localization of ROM error (using FOM dual solution)
        # tmp = -matrix_no_bc.dot(projected_reduced_solutions[i]) + rhs_no_bc[i]
        tmp = -matrix_no_bc.dot(projected_reduced_solutions[i]) + rhs_no_bc[i]
        if i > 0:
            # tmp -= jump_matrix_no_bc.dot(projected_reduced_solutions[i - 1])
            tmp -= jump_matrix_no_bc.dot(projected_reduced_solutions[i - 1])

        temporal_interval_error.append(np.dot(dual_solutions[i], tmp))
        temporal_interval_error_incidactor.append(0)
        # or  np.abs(temporal_interval_error[-1]/temporal_interval_error[i-1]):
            
        # if np.abs(temporal_interval_error[-1]) > tol:
        if np.abs(temporal_interval_error[-1])/np.dot(projected_reduced_solutions[i], dual_rhs_no_bc[i]) > tol_rel:
            temporal_interval_error_incidactor[-1] = 1
            pod_basis, space_time_pod_basis, reduced_system_matrix, reduced_jump_matrix, projected_reduced_solutions[-1], singular_values, total_energy = ROM_update(
                pod_basis, space_time_pod_basis, reduced_system_matrix, reduced_jump_matrix, projected_reduced_solutions[i - 1])
            #reduced_solutions[-1] = space_time_pod_basis.T.dot(projected_reduced_solutions[-1])
            reduced_solutions = space_time_pod_basis.T.dot(
                projected_reduced_solutions[-1])
            tmp = - \
                matrix_no_bc.dot(projected_reduced_solutions[i]) + rhs_no_bc[i]
            if i > 0:
                tmp -= jump_matrix_no_bc.dot(
                    projected_reduced_solutions[i - 1])
            temporal_interval_error[-1] = np.dot(dual_solutions[i], tmp)
            if np.abs(temporal_interval_error[-1]) > tol:
                print('Error correction failed')
        reduced_solutions_old = reduced_solutions

    end_execution = time.time()
    execution_time_ROM = end_execution - start_execution
    print("FOM time:   " + str(execution_time_FOM))
    print("ROM time:   " + str(execution_time_ROM))
    print("Size ROM:   " + str(pod_basis.shape[1]))
    print("FOM solves: " + str(np.sum(temporal_interval_error_incidactor)
                               ) + " / " + str(len(temporal_interval_error_incidactor)))
    projected_reduced_solution = np.hstack(projected_reduced_solutions)

    J_r = 0.
    J_r_t = np.empty([n_slabs,1])
    for i in range(n_slabs):
        J_r_t[i] = np.dot(projected_reduced_solutions[i], dual_rhs_no_bc[i])
        J_r += np.dot(projected_reduced_solutions[i], dual_rhs_no_bc[i])
    J["u_r"].append(J_r)

    print("J(u_h) =", J["u_h"])
    # TODO: in the future compare J(u_r) for different values of r
    print("J(u_r) =", J["u_r"][-1])
    # %% plot

    if PLOTTING:
        grid_t, grid_x = np.mgrid[0:4:100j, 0:1:100j]
        reduced_grid = scipy.interpolate.griddata(
            coordinates,
            projected_reduced_solution,
            (grid_t,
             grid_x),
            method=INTERPOLATION_TYPE)
        error_grid = scipy.interpolate.griddata(
            coordinates,
            np.abs(
                projected_reduced_solution -
                primal_solution),
            (grid_t,
             grid_x),
            method=INTERPOLATION_TYPE)
        fig, axs = plt.subplots(3)
        fig.suptitle(f"Projected reduced solution (ref={cycle.split('=')[1]})")
        # Plot 1: u_r
        im0 = axs[0].imshow(reduced_grid.T, extent=(
            0, 4, 0, 1), origin='lower')
        axs[0].set_xlabel("$t$")
        axs[0].set_ylabel("$x$")
        axs[0].set_title("$u_r$")
        fig.colorbar(im0, ax=axs[0])
        # Plot 2: u_h - u_r
        im1 = axs[1].imshow(error_grid.T, extent=(0, 4, 0, 1), origin='lower')
        axs[1].set_xlabel("$t$")
        axs[1].set_ylabel("$x$")
        axs[1].set_title("$u_h - u_r$")
        fig.colorbar(im1, ax=axs[1])

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
            cc += ['g']
        axs[2].plot(xx, yy)
        axs[2].plot(xx_FOM, yy_FOM, 'r')
        axs[2].set_xlabel("$t$")
        axs[2].set_ylabel("$\\eta$")
        axs[2].set_yscale("log")
        axs[2].set_title("temporal error estimate")
        plt.show()

        plt.plot(np.arange(0,n_slabs*time_step_size,time_step_size),J_r_t)
        plt.plot(np.arange(0,n_slabs*time_step_size,time_step_size),J_h_t,'r')
        plt.xlabel("$t$")
        plt.ylabel("$J(u)$")
        plt.title("temporal evaluation of cost funtional")
        plt.show()
        
        
    reduced_dual_matrix = dual_space_time_pod_basis.T.dot(
        matrix_no_bc.T.dot(dual_space_time_pod_basis))
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
