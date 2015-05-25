__author__ = 'giulio'

import numpy as np
import sys

sys.path.append("./../")

from utils.utils import indexing

def get_R(query_id):
    a = np.loadtxt("../data/qrels-treceval.txt", dtype='str')[:, 0].astype(int)
    R = np.count_nonzero(a == int(query_id))
    return R


def get_r_i_pseudo(mat_freq, rel_ids, query_term):
    mr = np.copy(mat_freq[rel_ids, :])
    r_i = np.count_nonzero(mr[:, int(query_term)])
    return r_i


def get_r_i(mat_freq, query_id, query_term):
    a = np.loadtxt("../data/qrels-treceval.txt", dtype='str')
    b = a[:, 0].astype(int)
    a = a[b == int(query_id)]
    a = a[:, 2].astype(int)
    a -= 1
    mr = np.copy(mat_freq[a, :])
    r_i = np.count_nonzero(mr[:, int(query_term)])
    return r_i


def p_pseudo(fm, q_id, pqw, pwords, dls, pj, res, k1, k2, b, avdl, N, R):
    # print pj
    actual_qw = []
    indexes_of_qws = []
    for qw in pqw:
        if qw in pwords:
            actual_qw.append(qw)
            indexes_of_qws.append(pwords[qw])

    indexes_of_qws = np.array(indexes_of_qws)
    tmp = np.arange(0, fm.shape[1])
    indexes_of_qws = np.in1d(tmp, indexes_of_qws)
    red_fm = fm[:, indexes_of_qws]
    idfs = np.ones(shape=(red_fm.shape[0], red_fm.shape[1]))
    tmp2 = np.copy(red_fm)
    tmp2[tmp2 != 0] = 1
    nis = tmp2.sum(axis=0)
    Ns = np.ones(red_fm.shape[1])*N
    idfs = np.log((Ns - nis + 0.5)/(nis + 0.5))
    Ks = k1*((1-b) + b*(dls/avdl))
    tf1s = red_fm*(k1 + 1)/(np.tile(Ks, (red_fm.shape[1], 1)).T + red_fm)
    tf2s = np.ones(red_fm.shape)
    ress = np.multiply(idfs, tf1s)
    ress = ress.sum(axis=1)
    idss = np.arange(0, red_fm.shape[0])
    res[pj, :, :] = np.vstack((idss, ress)).T

    ranking = res[pj, :, :]
    # R = 10
    indx = np.argsort(ranking[:, 1])[::-1][0:R]
    relevants = ranking[indx, :]
    rel_ids = relevants[:, 0].astype(int)
    r_is = []
    indexes_of_qws = []
    for qw in actual_qw:
        index_of_qw = pwords[qw]
        indexes_of_qws.append(pwords[qw])
        r_is.append(get_r_i_pseudo(fm, rel_ids, index_of_qw))

    r_is = np.array(r_is)
    indexes_of_qws = np.array(indexes_of_qws)

    tmp = np.arange(0, fm.shape[1])
    indexes_of_qws = np.in1d(tmp, indexes_of_qws)
    red_fm = fm[:, indexes_of_qws]
    idfs = np.ones(shape=(red_fm.shape[0], red_fm.shape[1]))
    tmp2 = np.copy(red_fm)
    tmp2[tmp2 != 0] = 1
    nis = tmp2.sum(axis=0)
    Ns = np.ones(red_fm.shape[1])*N
    # R = get_R(q_id)
    Rs = np.ones(red_fm.shape[1])*R
    r_is = np.array(r_is)
    idfs = np.log((N - nis - R + r_is + 0.5)*(r_is + 0.5)/(nis - r_is + 0.5)*(R - r_is + 0.5))
    Ks = k1*((1-b) + b*(dls/avdl))
    tf1s = red_fm*(k1 + 1)/(np.tile(Ks, (red_fm.shape[1], 1)).T + red_fm)
    tf2s = np.ones(red_fm.shape)
    ress = np.multiply(idfs, tf1s)
    # ress = np.multiply(ress, tf2s)
    ress = ress.sum(axis=1)
    idss = np.arange(0, red_fm.shape[0])

    idss_indx = np.argsort(ress)[::-1]

    idss = idss[idss_indx]
    ress = ress[idss_indx]
    res[pj, :, :] = np.vstack((idss, ress)).T


def p_esplicito(fm, q_id, pqw, pwords, dls, pj, res, k1, k2, b, avdl, N, R):
    # print pj
    actual_qw = []
    indexes_of_qws = []
    r_is = []
    for qw in pqw:
        if qw in pwords:
            actual_qw.append(qw)
            indexes_of_qws.append(pwords[qw])
            r_is.append(get_r_i(fm, q_id, pwords[qw]))

    indexes_of_qws = np.array(indexes_of_qws)
    tmp = np.arange(0, fm.shape[1])
    indexes_of_qws = np.in1d(tmp, indexes_of_qws)
    red_fm = fm[:, indexes_of_qws]
    idfs = np.ones(shape=(red_fm.shape[0], red_fm.shape[1]))
    tmp2 = np.copy(red_fm)
    tmp2[tmp2 != 0] = 1
    nis = tmp2.sum(axis=0)
    Ns = np.ones(red_fm.shape[1])*N
    R = get_R(q_id)
    Rs = np.ones(red_fm.shape[1])*R
    r_is = np.array(r_is)
    idfs = np.log((N - nis - R + r_is + 0.5)*(r_is + 0.5)/(nis - r_is + 0.5)*(R - r_is + 0.5))
    Ks = k1*((1-b) + b*(dls/avdl))
    tf1s = red_fm*(k1 + 1)/(np.tile(Ks, (red_fm.shape[1], 1)).T + red_fm)
    tf2s = np.ones(red_fm.shape)
    ress = np.multiply(idfs, tf1s)
    # ress = np.multiply(ress, tf2s)
    ress = ress.sum(axis=1)
    idss = np.arange(0, red_fm.shape[0])

    idss_indx = np.argsort(ress)[::-1]

    idss = idss[idss_indx]
    ress = ress[idss_indx]
    res[pj, :, :] = np.vstack((idss, ress)).T


def retrieve(k1, b, k2, R):
    freq_mat, docs_length, words = indexing()

    query_stem = np.loadtxt("../data/query-stem.txt", dtype='str', delimiter="\t ")

    queries = {}
    # print words
    for i, row in enumerate(query_stem):
        q_id = row[0]
        w = row[1]
        if q_id not in queries:
            queries[q_id] = [w]
        else:
            queries[q_id].append(w)
    #print queries
    N = freq_mat.shape[0]
    avdl = np.mean(docs_length)

    results = np.memmap("tmp", shape=(len(queries.keys()), N, 2), mode='w+', dtype='float')

    for pj, (q, lst) in enumerate(queries.iteritems()):
        p_pseudo(freq_mat, q, lst, words, docs_length, pj, results, k1, k2, b, avdl, N, R)

    # Parallel(n_jobs=cpu_count())(delayed(p_pseudo)(
    #    freq_mat, q, lst, words, docs_length, pj, results, k1, k2, b, avdl, N
    # ) for pj, (q, lst) in enumerate(queries.iteritems()))

    # considera e retrieva solo i documenti con uno score almeno threshold
    # aumento threshold, diminuisce percentuale di documenti rilevanti retrieved
    threshold = 0.0

    f = open('results.txt', 'w')
    for j, query in enumerate(results):
        stuff_toprint = results[j, 0:1000]
        stuff_toprint = stuff_toprint[stuff_toprint[:, 1] > threshold]
        for i, row in enumerate(stuff_toprint):
            f.write("%s Q0 %s %s %s G12R7\n" % (queries.keys()[j], int(row[0]+1), i+1, row[1]))
    f.close()

if __name__ == "__main__":
    k1 = float(sys.argv[1])
    b = float(sys.argv[2])
    k2 = float(sys.argv[3])
    R = float(sys.argv[4])
    retrieve(k1, b, k2, R)
