import numpy as np
import os
import matplotlib.pyplot as plt


PLOTTING = True
MOTHER_PATH = "/home/hendrik/Code/MORe_DWR/Elastodynamics/"
DATA_PATH = MOTHER_PATH + "/Data/3D/Rod/convergence_data"
DATA_PATH_DUAL = MOTHER_PATH + "Dual_Elastodynamics/Data/3D/Rod/"

LOAD_SOLUTION = False

stress = {'value': [], 'time': []}

for dir in sorted(os.listdir(DATA_PATH)):
    data = np.loadtxt(DATA_PATH + "/" + dir + "/stress_y.txt", delimiter=",")
    stress['time'].append(data[:,0])
    stress['value'].append(data[:,1])

error = []
for i in range(len(stress['value'])):
    error.append(np.linalg.norm(stress['value'][-1] - stress['value'][i]))

diff = []
for i in range(len(stress['value'])-1):
    diff.append(np.linalg.norm(stress['value'][i+1] - stress['value'][i]))


for i in range(len(stress['value'])):
    # print diff btw i+1 and i
    print("Diff to finest sol for " + str(i) + ": " + str(error[i]))
    if i < len(stress['value'])-2:
        print("        Convergence rate: " + str((error[i]/error[i+1])))
    

# name = diff[0] to string with 3 digits format





# plot stress list entries in one plot
if PLOTTING:
    plt.figure()
    for i in range(len(stress["time"])):
        plt.plot(stress['time'][i], stress['value'][i], label=str(i))
    plt.legend()
    plt.show()

