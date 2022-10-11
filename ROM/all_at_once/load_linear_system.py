import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import os

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
  
  # solution vector
  solution_cpp = np.loadtxt(OUTPUT_PATH + cycle + "/solution.txt")
  solution_py = scipy.sparse.linalg.spsolve(matrix_bc, rhs_bc)
  
  print("Linfty norms:")
  print("-------------")
  print("C++ solution:                                   ", np.linalg.norm(solution_cpp, ord=np.inf))
  print("Difference between C++ and Py solution:         ", np.linalg.norm(solution_cpp-solution_py, ord=np.inf))
  print("Relative difference between C++ and Py solution:", np.linalg.norm((solution_cpp-solution_py)/(solution_cpp+1e-19), ord=np.inf))
  
  print("\nJ(u):")
  print("-----")
  print("No BC:", np.dot(dual_rhs_no_bc, solution_cpp))
  print("BC:   ", np.dot(dual_rhs_bc, solution_cpp))