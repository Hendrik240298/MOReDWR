import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.interpolate
import matplotlib.pyplot as plt
import os

PLOTTING = True
INTERPOLATION_TYPE = "nearest" # "linear", "cubic"
OUTPUT_PATH = "../../FOM/all_at_once/output/dim=1/"

for cycle in os.listdir(OUTPUT_PATH):
  print(f"\n{'-'*12}\n| {cycle}: |\n{'-'*12}\n")
  
  # NO BC
  [data, row, column] = np.loadtxt(OUTPUT_PATH + cycle + "/matrix_no_bc.txt") 
  matrix_no_bc = scipy.sparse.csr_matrix((data, (row.astype(int), column.astype(int)))) 
  rhs_no_bc = np.loadtxt(OUTPUT_PATH + cycle + "/rhs_no_bc.txt")
  dual_rhs_no_bc = np.loadtxt(OUTPUT_PATH + cycle + "/dual_rhs_no_bc.txt")
  
  # BC
  [data, row, column] = np.loadtxt(OUTPUT_PATH + cycle + "/matrix_bc.txt") 
  matrix_bc = scipy.sparse.csr_matrix((data, (row.astype(int), column.astype(int))))
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
  
  # primal solution
  primal_solution = np.loadtxt(OUTPUT_PATH + cycle + "/solution.txt")
  if PLOTTING:
    grid_t, grid_x = np.mgrid[0:4:100j, 0:1:100j]
    primal_grid = scipy.interpolate.griddata(coordinates, primal_solution, (grid_t, grid_x), method=INTERPOLATION_TYPE)
    plt.title(f"Primal solution (ref={cycle.split('=')[1]})")
    plt.imshow(primal_grid.T, extent=(0,4,0,1), origin='lower')
    plt.xlabel("$t$")
    plt.ylabel("$x$")
    plt.colorbar()
    plt.show()
  
  # dual solution
  dual_solution = scipy.sparse.linalg.spsolve(dual_matrix, dual_rhs_bc) #matrix_bc.T
  if PLOTTING:
    grid_t, grid_x = np.mgrid[0:4:100j, 0:1:100j]
    dual_grid = scipy.interpolate.griddata(coordinates, dual_solution, (grid_t, grid_x), method=INTERPOLATION_TYPE)
    plt.title(f"Dual solution (ref={cycle.split('=')[1]})")
    plt.imshow(dual_grid.T, extent=(0,4,0,1), origin='lower')
    plt.xlabel("$t$")
    plt.ylabel("$x$")
    plt.colorbar()
    plt.show()
  
  # goal functionals
  J = {"u_h": np.dot(dual_rhs_no_bc, primal_solution), "u_r": []}
  print("J(u_h) =", J["u_h"])
  
  # for different POD sizes, do POD-ROM and calculate error in goal functional
  # TODO: primal POD-ROM
  # TODO: evaluate J(u_h) - J(u_r)
  # TODO: project dual solution onto first r POD vectors (?)
  # TODO: evaluate error estimator
  