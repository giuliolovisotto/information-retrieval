__author__ = 'giulio'

import subprocess
from get_map import *
import numpy as np

dim = 21

params = np.zeros(shape=(dim, 4))

maps = np.zeros(dim)

params[:, 0] = 0.1  # k1
params[:, 1] = 0.0  # b
params[:, 2] = 1.0  # k2
params[:, 3] = np.linspace(0.0, 1.0, dim)  # alpha


for i in range(params.shape[0]):
    subprocess.call([
        'python', '../lab5/main.py', str(params[i, 0]), str(params[i, 1]), str(params[i, 2]), str(params[i, 3])
    ])
    maps[i] = evaluate_map("results.txt")
    print params[i, 3], maps[i]

# header = b, map

np.savetxt("lab5_plots_alpha.txt", np.vstack((params[:, 3], maps)).T, delimiter=',', header="alpha,map", comments='')
