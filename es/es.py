__author__ = 'giulio'

import numpy as np
from subprocess import call
from get_map import evaluate_map
from joblib import Parallel, cpu_count, delayed
import sys

sys.path.append("./../")

from utils.utils import indexing

init_f = 1.0
ps = 10
lambda_v = 35

# best for now
# [0.02533949  0.02218594]
# 0.3354

lb = np.array([1e-8, 1e-8, 0.0])
ub = np.array([2.0, 1.0, 1.0])

N = 3

start = np.array([1.2, 0.75, 0.5])

X = np.tile(start, (ps, 1))

S = np.tile(np.array([init_f, init_f, init_f]), (10, 1)) * (X/np.sqrt(N))

tau = 1/np.sqrt(2 * N)
tau2 = 1/np.sqrt(2 * np.sqrt(N))

fit = np.ones(shape=(ps, 1)) * 30.00

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
    call(["python", "../lab5/main.py", str(x[0]), str(x[1]), '1.0', str(x[2])])
    out[j] = evaluate_map("results.txt")


def evaluate(pop):
    # fitnesses = np.zeros(shape=(pop.shape[0]), dtype='float')
    fitnesses = np.memmap("tmp2", shape=(pop.shape[0]), dtype='float', mode='w+')
    n_j = cpu_count()

    for ii, x in enumerate(pop):
        sys.stdout.write("\r%s/%s" % (ii+1, pop.shape[0]))
        single_eval(x, str(ii % n_j), fitnesses, ii)

    # Parallel(n_jobs=n_j)(delayed(single_eval)(x, str(ii % n_j), fitnesses, ii) for ii, x in enumerate(pop))

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
