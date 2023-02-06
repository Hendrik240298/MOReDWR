import numpy as np
import matplotlib.pyplot as plt

# read in data
python_data = np.loadtxt('python_goal_func_error.txt', delimiter=',')
cpp_data = np.loadtxt('cpp_goal_func_error.txt', delimiter=',')

plt.rcParams['text.usetex'] = True

# create a figure with three subplots
fig, axes = plt.subplots(3, 1)
fig.suptitle(r"$J(U_h) - J(U_r)$", fontsize=16)

axes[0].set_title("Py")
axes[0].plot(python_data[:,0], python_data[:,1], label='python')

axes[1].set_title("C++")
axes[1].plot(cpp_data[:, 0], cpp_data[:, 1], label='cpp')

# axes[2].plot(python_data[:,0], python_data[:,1], label='python')
# axes[2].plot(cpp_data[:, 0], cpp_data[:, 1], label='cpp')
# axes[2].legend()
axes[2].set_title("C++ divided by Py")
axes[2].plot(cpp_data[:, 0], cpp_data[:, 1] / python_data[:, 1], label='cpp')
plt.show()