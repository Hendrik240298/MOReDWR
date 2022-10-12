import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.interpolate
import matplotlib.pyplot as plt
import os

PLOTTING = True
INTERPOLATION_TYPE = "nearest"  # "linear", "cubic"
OUTPUT_PATH = "../../FOM/all_at_once/output/dim=1/"

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
    primal_solution = np.loadtxt(OUTPUT_PATH + cycle + "/solution.txt")
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
    print("J(u_h) =", J["u_h"])

    # -------------- #
    # PRIMAL POD-ROM #
    # -------------- #
    print(f"#DoFs(space): {n_dofs['space']}")
    print(f"#DoFs(time): {n_dofs['time']}")

    # ---------------
    # snapshot matrix
    Y = primal_solution.reshape(n_dofs["time"], n_dofs["space"]).T

    # ------------------
    # correlation matrix
    # TODO: include mass matrix
    correlation_matrix = np.dot(Y.T, Y)
    assert correlation_matrix.shape == (
        n_dofs["time"], n_dofs["time"]), f"Matrix should be of #DoFs(time) shape"

    # -----------------------------------------
    # compute eigenvalues of correlation matrix
    eigen_values, eigen_vectors = np.linalg.eig(correlation_matrix)

    # ignore the complex part of the eigen values and eigen vectors
    eigen_values, eigen_vectors = eigen_values.astype(
        'float64'), eigen_vectors.astype('float64')

    # sort the eigenvalues in descending order
    eigen_values, eigen_vectors = eigen_values[eigen_values.argsort(
    )[::-1]], eigen_vectors[eigen_values.argsort()[::-1]]

    # plot eigen value decay
    if PLOTTING:
        plt.plot(
            range(1, min(n_dofs["space"], n_dofs["time"])+1), eigen_values)
        plt.yscale("log")
        plt.xlabel("index")
        plt.ylabel("eigen value")
        plt.title("Eigenvalues of correlation matrix")
        plt.show()

    # desired energy ratio
    ENERGY_RATIO_THRESHOLD = 0.999

    # determine number of POD basis vectors needed to preserve certain ratio of energy
    r = np.sum([(eigen_values[:i].sum() / eigen_values.sum()) <
               ENERGY_RATIO_THRESHOLD for i in range(n_dofs["time"])])
    print(
        f"To preserve {ENERGY_RATIO_THRESHOLD} of information we need {r} POD vector(s). (result: {eigen_values[:r].sum() / eigen_values.sum()} information).")

    # -------------------------
    # compute POD basis vectors
    pod_basis = np.dot(np.dot(Y, eigen_vectors[:, :r]), np.diag(
        1. / np.sqrt(eigen_values[:r])))

    # plot the POD basis
    if PLOTTING:
        plt.plot(coordinates_x, pod_basis[:, 0], color="blue")
        #plt.plot(coordinates_x, pod_basis[:, 1], color="red")
        plt.xlabel("$x$")
        plt.title("POD vectors")
        plt.show()

    # change from the FOM to the POD basis
    reduced_system_matrix = scipy.sparse.dok_matrix(
        (r*n_dofs["time"], r*n_dofs["time"]), dtype=np.float32)
    reduced_rhs = np.zeros(r*n_dofs["time"], dtype=np.float32)

    #system_matrix_block = matrix_no_bc.tobsr(blocksize=(n_dofs["space"],n_dofs["space"]),copy=False) -> rather e.g. 2 * n_dofs["space"] as block-size

    # TODO: dow we want to use bsr_matrices?!
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

    # NOTE: old space reduction
    reduced_system_matrix = np.dot(
        np.dot(pod_basis.T, system_matrix.array()), pod_basis)
    reduced_rhs_matrix = np.dot(
        np.dot(pod_basis.T,    rhs_matrix.array()), pod_basis)

    # TODO: primal POD-ROM
    # TODO: evaluate J(u_h) - J(u_r)
    # TODO: project dual solution onto first r POD vectors (?)
    # TODO: evaluate error estimator
