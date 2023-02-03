import numpy as np
import matplotlib.pyplot as plt

# read in data
python_data = np.loadtxt('python_goal_func_error.txt', delimiter=',')
error_estimator_data = np.loadtxt('dealii_estimator_error.txt', delimiter=',')

plt.rcParams['text.usetex'] = True

# create a figure with two subplots
fig, axes = plt.subplots(2, 1)
fig.suptitle("Comparison QoI error and estimator", fontsize=16)

axes[0].set_title("$J(U_h) - J(U_r)$")
axes[0].plot(python_data[:,0], python_data[:,1], label='J(error)')

axes[1].set_title(r"$\\\eta$")
axes[1].plot(error_estimator_data[:, 0], error_estimator_data[:, 1], label='eta')

# axes[2].plot(python_data[:,0], python_data[:,1], label='python')
# axes[2].plot(cpp_data[:, 0], cpp_data[:, 1], label='cpp')
# axes[2].legend()
# axes[2].set_title("C++ divided by Py")
# axes[2].plot(cpp_data[:, 0], cpp_data[:, 1] / python_data[:, 1], label='cpp')
plt.show()
