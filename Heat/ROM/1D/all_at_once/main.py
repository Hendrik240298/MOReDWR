import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.linalg
import scipy.interpolate
import matplotlib.pyplot as plt
import os

PLOTTING = False #True
INTERPOLATION_TYPE = "nearest"  # "linear", "cubic"
LOAD_PRIMAL_SOLUTION = False
OUTPUT_PATH = "../../../Data/1D/moving_source/all_at_once/"


def iPOD(POD, bunch, singular_values, snapshot, total_energy):
    bunch_size = 2
    energy_content = 0.9999
    row, col_POD = POD.shape
    bunch = np.hstack((bunch, snapshot.reshape(-1, 1)))
    _, col = bunch.shape

    total_energy += np.dot(snapshot, snapshot)

    if (col == bunch_size):
        if (col_POD == 0):
            POD, S, _ = scipy.linalg.svd(bunch, full_matrices=False)
            r = 0
            # np.shape(S_k)[0])):
            while ((np.dot(S[0:r], S[0:r])/total_energy <= energy_content) and (r <= bunch_size)):
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

            if (np.linalg.norm(np.matmul(Q_q.T, Q_q)-np.eye(np.shape(Q_q)[1])) >= 1e-14):
                Q_q, R_q = scipy.linalg.qr(Q_q, mode='economic')
                K = np.matmul(R_q, K)

            U_k, S_k, _ = scipy.linalg.svd(K, full_matrices=False)
            Q = np.hstack((POD, Q_p))
            Q_q, _ = scipy.linalg.qr(Q, mode='economic')

            r = 0
            while ((np.dot(S_k[0:r], S_k[0:r])/total_energy <= energy_content) and (r <= np.shape(S_k)[0])):
                r += 1
                # print(r)

            singular_values = S_k[0:r]
            POD = np.matmul(Q_q, U_k[:, 0:r])
        bunch = np.empty([np.shape(bunch)[0], 0])
        # print(np.shape(POD))

    return POD, bunch, singular_values, total_energy


for cycle in os.listdir(OUTPUT_PATH):
    print(f"\n{'-'*12}\n| {cycle}: |\n{'-'*12}\n")

    # NO BC
    [data, row, column] = np.loadtxt(OUTPUT_PATH + cycle + "/matrix_no_bc.txt")
    matrix_no_bc = scipy.sparse.csr_matrix(
        (data, (row.astype(int), column.astype(int))))
    rhs_no_bc = np.loadtxt(OUTPUT_PATH + cycle + "/rhs_no_bc.txt")
    dual_rhs_no_bc = np.loadtxt(OUTPUT_PATH + cycle + "/dual_rhs_no_bc.txt")

    # BC
    [data, row, column] = np.loadtxt(OUTPUT_PATH + cycle + "/matrix_bc.txt")
    matrix_bc = scipy.sparse.csr_matrix(
        (data, (row.astype(int), column.astype(int))))
    rhs_bc = np.loadtxt(OUTPUT_PATH + cycle + "/rhs_bc.txt")
    dual_rhs_bc = np.loadtxt(OUTPUT_PATH + cycle + "/dual_rhs_bc.txt")

    # applying BC to dual matrix
    dual_matrix = matrix_no_bc.T.tocsr()
    for row in set((matrix_no_bc-matrix_bc).nonzero()[0]):
        for col in dual_matrix.getrow(row).nonzero()[1]:
            dual_matrix[row, col] = 1. if row == col else 0.

    # coordinates
    coordinates_x = np.loadtxt(OUTPUT_PATH + cycle + "/coordinates_x.txt")
    coordinates_t = np.loadtxt(OUTPUT_PATH + cycle + "/coordinates_t.txt")
    coordinates = np.vstack((
        np.tensordot(coordinates_t, np.ones_like(coordinates_x), 0).flatten(),
        np.tensordot(np.ones_like(coordinates_t), coordinates_x, 0).flatten()
    )).T
    n_dofs = {"space": coordinates_x.shape[0], "time": coordinates_t.shape[0]}

    # primal solution
    primal_solution = np.loadtxt(
        OUTPUT_PATH + cycle + "/solution.txt") if LOAD_PRIMAL_SOLUTION else scipy.sparse.linalg.spsolve(matrix_bc, rhs_bc)
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

    # dual solution
    dual_solution = scipy.sparse.linalg.spsolve(dual_matrix, dual_rhs_bc)
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

    # goal functionals
    J = {"u_h": np.dot(dual_rhs_no_bc, primal_solution), "u_r": []}

    # -------------- #
    # PRIMAL POD-ROM #
    # -------------- #
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

    # pod_basis_svd = iPOD(pod_basis,pod_basis,Y[:,0])
    total_energy = 0
    POD = np.empty([0, 0])
    bunch = np.empty([np.shape(Y)[0], 0])
    singular_values = np.empty([0, 0])
    for i in range(1, np.shape(Y)[1], 1):
        POD, bunch, singular_values, total_energy = iPOD(
            POD, bunch, singular_values, Y[:, i], total_energy)
    POD_full, singular_values_test, _ = scipy.linalg.svd(Y, full_matrices=False)
    POD_full = POD_full[:,0:np.shape(singular_values)[0]]
    # plot eigen value decay
    if PLOTTING:
        plt.plot(
            range(1, min(n_dofs["space"], n_dofs["time"])+1), eigen_values, label="correlation matrix")
        plt.plot(
            range(1, np.shape(singular_values)[0]+1), np.power(singular_values,2), label="iSVD")
        plt.plot(
            range(1, np.shape(singular_values)[0]+1), np.power(singular_values_test[0:np.shape(singular_values)[0]],2), label="SVD")
        plt.legend()
        plt.yscale("log")
        plt.xlabel("index")
        plt.ylabel("eigen value")
        plt.title("Eigenvalues of correlation matrix")
        plt.show()

        plt.plot(
            range(1, min(n_dofs["space"], n_dofs["time"])+1), dual_eigen_values)
        plt.yscale("log")
        plt.xlabel("index")
        plt.ylabel("eigen value")
        plt.title("Dual eigenvalues of correlation matrix")
        plt.show()

    # desired energy ratio
    ENERGY_RATIO_THRESHOLD = 0.9999
    ENERGY_RATIO_THRESHOLD_DUAL = 0.9999999

    # determine number of POD basis vectors needed to preserve certain ratio of energy
    r = np.sum([(eigen_values[:i].sum() / eigen_values.sum()) <
               ENERGY_RATIO_THRESHOLD for i in range(n_dofs["time"])])
    # FOR DEBUGGING: r = 2
    r_dual = np.sum([(dual_eigen_values[:i].sum() / dual_eigen_values.sum()) <
                     ENERGY_RATIO_THRESHOLD_DUAL for i in range(n_dofs["time"])])

    print(
        f"To preserve {ENERGY_RATIO_THRESHOLD} of information we need {r} primal POD vector(s). (result: {eigen_values[:r].sum() / eigen_values.sum()} information).")
    print(
        f"To preserve {ENERGY_RATIO_THRESHOLD} of information we need {POD.shape[1]} primal iPOD vector(s).")
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
            plt.plot(
                coordinates_x, pod_basis[:, i], color=colors[i % len(colors)], label=str(i+1))
        plt.legend()
        plt.xlabel("$x$")
        plt.title("Primal POD vectors")
        plt.show()

        for i in range(r_dual):
            plt.plot(coordinates_x, dual_pod_basis[:, i], color=colors[i % len(
                colors)], label=str(i+1))
        plt.legend()
        plt.xlabel("$x$")
        plt.title("Dual POD vectors")
        plt.show()

    # change from the FOM to the POD basis
    space_time_pod_basis = scipy.sparse.block_diag(
        [pod_basis]*n_dofs["time"])  # NOTE: this might be a bit inefficient
    dual_space_time_pod_basis = scipy.sparse.block_diag(
        [dual_pod_basis]*n_dofs["time"])
    # if PLOTTING:
    #plt.spy(space_time_pod_basis, markersize=2)
    # plt.show()

    reduced_system_matrix = space_time_pod_basis.T.dot(
        matrix_no_bc.dot(space_time_pod_basis))
    reduced_rhs = space_time_pod_basis.T.dot(rhs_no_bc)

    reduced_dual_matrix = dual_space_time_pod_basis.T.dot(
        matrix_no_bc.T.dot(dual_space_time_pod_basis))
    reduced_dual_rhs = dual_space_time_pod_basis.T.dot(dual_rhs_no_bc)

    # solve reduced linear system
    reduced_solution = scipy.sparse.linalg.spsolve(
        reduced_system_matrix, reduced_rhs)
    reduced_dual_solution = scipy.sparse.linalg.spsolve(
        reduced_dual_matrix, reduced_dual_rhs)
    # print(reduced_solution.shape)
    #true_reduced_solution = space_time_pod_basis.T.dot(primal_solution)
    # print(true_reduced_solution.reshape(-1,r)[:,0:1])
    # print(scipy.sparse.block_diag([pod_basis[:,0:1]]*n_dofs["time"]).T.dot(primal_solution))
    # print(space_time_pod_basis.shape)
    # print(true_reduced_solution.shape)
    #print(f"Reduced solution is wrong by a factor of {reduced_solution[0]/true_reduced_solution[0]}")

    projected_reduced_solution = space_time_pod_basis.dot(reduced_solution)
    projected_reduced_dual_solution = dual_space_time_pod_basis.dot(
        reduced_dual_solution)
    J["u_r"].append(np.dot(dual_rhs_no_bc, projected_reduced_solution))
    print("J(u_h) =", J["u_h"])
    # TODO: in the future compare J(u_r) for different values of r
    print("J(u_r) =", J["u_r"][-1])

    if PLOTTING:
        grid_t, grid_x = np.mgrid[0:4:100j, 0:1:100j]
        reduced_grid = scipy.interpolate.griddata(
            coordinates, projected_reduced_solution, (grid_t, grid_x), method=INTERPOLATION_TYPE)
        error_grid = scipy.interpolate.griddata(coordinates, np.abs(
            projected_reduced_solution-primal_solution), (grid_t, grid_x), method=INTERPOLATION_TYPE)
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
        # temporal localization of ROM error (using FOM dual solution)
        temporal_interval_error = []
        tmp = -matrix_no_bc.dot(projected_reduced_solution) + rhs_no_bc
        # WARNING: hardcoding dG(1) time discretization here
        dofs_per_time_interval = 2 * n_dofs["space"]
        i = 0
        while (i != n_dofs["time"] * n_dofs["space"]):
            temporal_interval_error.append(
                dual_solution[i:i +
                              dofs_per_time_interval].dot(tmp[i:i+dofs_per_time_interval])
            )
            i += dofs_per_time_interval
        # WARNING: hardcoding end time T = 4.
        time_step_size = 4.0 / (n_dofs["time"] / 2)
        xx, yy = [], []
        for i, error in enumerate(temporal_interval_error):
            xx += [i * time_step_size,
                   (i+1) * time_step_size, (i+1) * time_step_size]
            yy += [abs(error), abs(error), np.inf]
        axs[2].plot(xx, yy)
        axs[2].set_xlabel("$t$")
        axs[2].set_ylabel("$\eta$")
        axs[2].set_yscale("log")
        axs[2].set_title("temporal error estimate")
        plt.show()

    # ----------------
    # error estimation
    true_error = J['u_h']-J['u_r'][-1]

    # using FOM dual solution
    print("\nUsing z_h:")
    print("----------")
    error_estimator = -dual_solution.dot(matrix_no_bc.dot(
        projected_reduced_solution)) + dual_solution.dot(rhs_no_bc)
    print(f"True error:        {true_error}")
    print(f"Estimated error:   {error_estimator}")
    print(f"Effectivity index: {abs(true_error / error_estimator)}")

    # using ROM dual solution
    print("\nUsing z_r:")
    print("----------")
    error_estimator_cheap = -projected_reduced_dual_solution.dot(matrix_no_bc.dot(
        projected_reduced_solution)) + projected_reduced_dual_solution.dot(rhs_no_bc)
    #error_estimator_cheap = 0.5 * error_estimator_cheap + 0.5 * (-projected_reduced_solution.dot(matrix_no_bc.T.dot(projected_reduced_dual_solution)) + projected_reduced_solution.dot(dual_rhs_no_bc))
    #error_estimator_cheap = 0.5 * error_estimator_cheap + 0.5 * (-primal_solution.dot(matrix_no_bc.T.dot(projected_reduced_dual_solution)) + primal_solution.dot(dual_rhs_no_bc))
    print(f"True error:              {true_error}")
    print(f"Cheap estimated error:   {error_estimator_cheap}")
    print(
        f"Effectivity index:       {abs(true_error / error_estimator_cheap)}")

    """
    reduced_system_matrix = scipy.sparse.dok_matrix(
        (r*n_dofs["time"], r*n_dofs["time"]), dtype=np.float32)
    reduced_rhs = np.zeros(r*n_dofs["time"], dtype=np.float32)

    #system_matrix_block = matrix_no_bc.tobsr(blocksize=(n_dofs["space"],n_dofs["space"]),copy=False) -> rather e.g. 2 * n_dofs["space"] as block-size

    # TODO: do we want to use bsr_matrices?!
    # TODO: parallel reduction of ST-system matrix can be done similar to parallel assembly in https://github.com/mathmerizing/MultigridPython/blob/master/assembly.py

    plt.spy(matrix_no_bc, markersize=2)
    plt.show()

    # TODO: reduction of space-time system matrix
    # TODO: think about the fact that blocks are e.g. of size 2 * n_dofs["space"] --> smarter way would be to consider temporal sparsity pattern and then reduce for each entry in temporal sparsity pattern
    primal_nonzero = matrix_no_bc.nonzero()
    index = 0
    for ii in range(n_dofs["time"]):
        matrix_time_dof = scipy.sparse.dok_matrix(
            (n_dofs["space"], n_dofs["space"]), dtype=np.float32)
        matrix_jump = scipy.sparse.dok_matrix(
            (n_dofs["space"], n_dofs["space"]), dtype=np.float32)

        x = primal_nonzero[0][index]
        while x < (ii+1)*n_dofs["space"]:
            y = primal_nonzero[1][index]
            if y >= ii * n_dofs["space"]:
                print(x-ii*n_dofs["space"])
                print(y-ii*n_dofs["space"])
                matrix_time_dof[x-ii*n_dofs["space"],y-ii*n_dofs["space"]] = matrix_no_bc[x, y]
            else:
                matrix_jump[x-ii*n_dofs["space"],y-(ii-1)*n_dofs["space"]] = matrix_no_bc[x, y]
            index += 1
    """
