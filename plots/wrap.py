__author__ = 'giulio'

import subprocess
from get_map import *
import numpy as np

dim = 10

params = np.zeros(shape=(dim, 7))

maps = np.zeros(dim)

params[:, 0] = 0.02  # k1
params[:, 1] = 0.02  # b
params[:, 2] = 1.0  # k2
params[:, 3] = np.linspace(10, 100, dim)  # Nd
params[:, 4] = 1.0  # alpha
params[:, 5] = 0.1  # beta
params[:, 6] = 0.1  # gamma



for i in range(params.shape[0]):
    subprocess.call([
        'python', '../lab7/main.py', str(params[i, 0]), str(params[i, 1]), str(params[i, 2]), str(params[i, 3]),
        str(params[i, 4]), str(params[i, 5]), str(params[i, 6]),
    ])
    maps[i] = evaluate_map("results.txt")
    print params[i, 3], maps[i]

# header = b, map

np.savetxt("lab7_plots_N.txt", np.vstack((params[:, 3], maps)).T, delimiter=',', header="alpha,map", comments='')
