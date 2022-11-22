import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.linalg
import scipy.interpolate
import matplotlib.pyplot as plt
import os
import time
import sys
# %%
DATA_PATH = '../../../Data/2D/rotating_circle/slabwise/cycle=6'

n_dofs = {"space": 16641, "time": 256*2}
primal_solutions = []
n_slabs = 256
for i in range(n_slabs):
    primal_solutions.append(np.loadtxt(DATA_PATH + f"/solution_{(5-len(str(i)))*'0'}{i}.txt"))
    
# %%
primal_solution = np.hstack(primal_solutions)
Y = primal_solution.reshape(n_dofs["time"], n_dofs["space"]).T

pod_basis, singular_values, _= scipy.linalg.svd(Y, full_matrices=False)

plt.plot(range(1,min(n_dofs["space"],n_dofs["time"]) + 1), singular_values)
plt.legend()
plt.yscale("log")
plt.xlabel("index")
plt.ylabel("eigen value")
plt.title("sigs")
plt.show()

# %%
energy = []
for i in range(len(singular_values)):
    energy.append(np.sum(np.power(singular_values[range(i)],2))/
                  np.sum(np.power(singular_values,2)))
    

plt.plot(range(1,min(n_dofs["space"],n_dofs["time"]) + 1), [1-e for e in energy])
plt.legend()
plt.yscale("log")
plt.xlabel("index")
plt.ylabel("energy loss")
plt.title("energy loss")
plt.show()