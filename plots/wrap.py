__author__ = 'giulio'

import subprocess
from get_map import *
import numpy as np

dim = 10

params = np.zeros(shape=(dim, 4))

maps = np.zeros(dim)

params[:, 0] = 0.1
params[:, 1] = 0.0
params[:, 2] = 1.0
params[:, 3] = np.linspace(5, 50, dim)


for i in range(params.shape[0]):
    subprocess.call(['python', '../lab4/main.py', str(params[i, 0]), str(params[i, 1]), str(params[i, 2]),
                     str(params[i, 3])])
    maps[i] = evaluate_map("results.txt")
# header = b, map

print np.vstack((params[:, 3], maps)).T

# np.savetxt("lab3_plots_b.txt", np.vstack((params[:, 1], maps)).T, delimiter=',', header="b,map", comments='')
