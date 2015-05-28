__author__ = 'giulio'

import numpy as np
from subprocess import call
from get_map import evaluate_map
from joblib import Parallel, cpu_count, delayed
import sys

sys.path.append("./../")

from utils.utils import indexing

_python_int = "python2"

init_f = 0.5
ps = 10
lambda_v = 35

# best for now (lab3)
# partito da [1.2, 0.75]
# [0.02533949  0.02218594] k1, b
# 0.3354

# best for now (lab4) esplicito
# partito da [1.2, 0.75]
# [2.06449110e-02   2.25921442e-05] R=50
# 0.3504

# best for now (lab4) pseudo
# partito da [1.2, 0.75] R=50
# [ 2.02647912e-02   1.00000000e-08] R=50
# 0.3022

# best for now (lab5)
# partito da [1.2, 0.75, 0.5]
# [2.36851802e-02   1.00000000e-08   7.83202800e-01]
# 0.3366

# best for now (lab6) N=10, m=2
# partito da [1.2, 0.75]
# [8.70400657e-03   1.00000000e-08]
# 0.2648

# best for now (lab7)
# partito da [1.2, 0.75, 1.0, 0.1, 0.1]
# [0.02466756  0.01853281  1.         0.10971746  0.05485873]
# 0.3405

lb = np.array([1e-8, 1e-8])
ub = np.array([2.0, 1.0])

N = 2

start = np.array([1.2, 0.75])

X = np.tile(start, (ps, 1))

S = np.tile(np.array(np.tile(init_f, (1, N))), (10, 1)) * (X/np.sqrt(N))

tau = 1/np.sqrt(2 * N)
tau2 = 1/np.sqrt(2 * np.sqrt(N))

call([_python_int, "../lab6/main.py", str(start[0]), str(start[1]), '1.0', '10'])
fit = evaluate_map("results.txt")

fitn = np.ones(shape=(ps, )) * fit

f = open('feval.txt', 'a')
f.write("%s,%s\n" % (0, fitn[0]))
f.close()


def discrete_recombination(pop, l):
    r_pop = np.zeros((l, pop.shape[1]))
    for j in range(l):
        rand_elem_x = np.random.randint(0, pop.shape[0], size=(pop.shape[1],))
        for k in range(pop.shape[1]):
            r_pop[j, k] = pop[rand_elem_x[k], k]
    return r_pop

def n_mutation(pop, sigmas, tau, tau2, lb, ub):
    m_pop = np.zeros(shape=(pop.shape[0], pop.shape[1]))
    m_sigm = np.zeros(shape=(pop.shape[0], pop.shape[1]))
    for i in range(pop.shape[0]):
        my_r = np.random.randn()
        m_sigm[i, :] = sigmas[i, :] * np.exp(tau * my_r + tau2 * np.random.randn(N))
        m_pop[i, :] = pop[i, :] + np.multiply(m_sigm[i, :], np.random.randn(N))
        m_pop[i, :] = np.minimum(np.maximum(m_pop[i, :], lb), ub)
    return m_pop, m_sigm

def single_eval(x, ind, out, j):
    call([_python_int, "../lab6/main.py", str(x[0]), str(x[1]), '1.0', '10'])
    out[j] = evaluate_map("results.txt")


def evaluate(pop):
    # fitnesses = np.zeros(shape=(pop.shape[0]), dtype='float')
    fitnesses = np.memmap("tmp2", shape=(pop.shape[0]), dtype='float', mode='w+')
    n_j = 4

    for ii, x in enumerate(pop):
        sys.stdout.write("\r%s/%s" % (ii+1, pop.shape[0]))
        sys.stdout.flush()
        single_eval(x, str(ii % n_j), fitnesses, ii)

    # Parallel(n_jobs=n_j)(delayed(single_eval)(x, ii, fitnesses, ii) for ii, x in enumerate(pop))

    return fitnesses

def comma_selection(mu, pop, sigmas, fevals):
    indx = np.argsort(fevals)[::-1]
    new_pop = (pop[indx])[:mu]
    new_sigm = (sigmas[indx])[:mu]
    new_f = (fevals[indx])[:mu]
    return new_pop, new_sigm, new_f


for i in range(1000):
    print "iteration %s" % str(i+1)
    X = discrete_recombination(X, lambda_v)
    S = discrete_recombination(S, lambda_v)

    X, S = n_mutation(X, S, tau, tau2, lb, ub)

    fitn = evaluate(X)
    X, S, fitn = comma_selection(ps, X, S, fitn)
    print fitn, X[0, :]
    f = open('feval.txt', 'a')
    f.write("%s,%s\n" % (i, fitn[0]))
    f.close()
