import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.linalg
import scipy.interpolate
from scipy.sparse import coo_matrix, bmat
import matplotlib.pyplot as plt
from auxiliaries import save_vtk, read_in_LES, apply_boundary_conditions, read_in_discretization,solve_primal_FOM_step, solve_dual_FOM_step, solve_primal_ROM_step, reorder_matrix,reorder_vector,error_estimator,save_solution_txt, load_solution_txt, evaluate_cost_functional


MOTHER_PATH = "/home/hendrik/Code/MORe_DWR/Elastodynamics/"
MOTHER_PATH = "/home/ifam/roth/Desktop/Code/dealii_dir/iROM/MORe_DWR/Elastodynamics/"
OUTPUT_PATH = MOTHER_PATH + "/Data/3D/Rod/"
OUTPUT_PATH_DUAL = MOTHER_PATH + "Dual_Elastodynamics/Data/3D/Rod/"
cycle = "cycle=1"
SAVE_PATH = MOTHER_PATH + "Data/ROM/" + cycle + "/"

# %% Reading in matricies and rhs without bc
matrix_no_bc, _ = read_in_LES(OUTPUT_PATH + cycle, "/matrix_no_bc.txt", "primal_rhs_no_bc")
mass_matrix_no_bc, _ = read_in_LES(OUTPUT_PATH + cycle, "/mass_matrix_no_bc.txt", "primal_rhs_no_bc")
system_matrix, _ = read_in_LES(OUTPUT_PATH_DUAL + cycle, "/dual_matrix_no_bc.txt", "dual_rhs_no_bc")

primal_matrix_transposed_in_dual, _ = read_in_LES(OUTPUT_PATH_DUAL + cycle, "/matrix_no_bc.txt", "dual_rhs_no_bc")

# %% plot matricies
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
ax1 = axs[0, 0]
ax2 = axs[0, 1]
ax3 = axs[1, 0]
ax4 = axs[1, 1]

ax1.spy(system_matrix, precision=1e-14)
ax1.set_title("Dual system matrix")

ax2.spy(primal_matrix_transposed_in_dual, precision=1e-14)
ax2.set_title("Dual primal matrix assemled tranposed")

ax3.spy(matrix_no_bc.T, precision=1e-14)
ax3.set_title("Primal system matrix transposed")

ax4.spy(mass_matrix_no_bc.T, precision=1e-14)
ax4.set_title("Primal mass matrix transposed")

plt.show()


# %% plot diff in matricies

mat = np.abs(((mass_matrix_no_bc.T + matrix_no_bc.T)-system_matrix).todense())

max_value = np.max(np.max(mat))
# prepare x and y for scatter plot
plot_list = []
for rows,cols in zip(np.where(mat!=0)[0],np.where(mat!=0)[1]):
    if (np.abs(mat[rows,cols]) > 1e-8): #-8):
        plot_list.append([cols,rows,mat[rows,cols]])
for i in range(3*702):
    plot_list.append([i,351,max_value])
    plot_list.append([351,i,max_value])
    plot_list.append([i,351+350,max_value])
    plot_list.append([351+350,i,max_value])
    plot_list.append([i,351+2*350,max_value])
    plot_list.append([351+2*350,i,max_value])
plot_list = np.array(plot_list)

# scatter plot with color bar, with rows on y axis
plt.scatter(plot_list[:,0],plot_list[:,1],c=plot_list[:,2], s=1)

cb = plt.colorbar()
# cb colorbar logarithmic
#cb.set_ticks([1e-7,1e-6,1e-5,1e-4,1e-3,1e-2])

# full range for x and y axes
plt.xlim(0,mat.shape[1])
plt.ylim(0,mat.shape[0])
# invert y axis to make it similar to imshow
plt.gca().invert_yaxis()

#ax3.set_title("Difference")

plt.show()