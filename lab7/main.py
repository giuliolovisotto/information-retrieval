__author__ = 'giulio'

import numpy as np
import networkx as nx
import sys

sys.path.append("./../")

from utils.utils import indexing


def get_graph_N(root_nodes):
    G = nx.read_edgelist("../data/citation_n.txt", delimiter=" ", create_using=nx.DiGraph())
    R = map(str, root_nodes)
    E = []

    for n in R:
        for source, dest in G.in_edges(n):
            E.append(source)
        for source, dest in G.out_edges(n):
            E.append(dest)

    E = map(str, list(np.unique(E)))
    B = set(R + E)
    N = set(G.nodes())
    to_remove = N - B
    G.remove_nodes_from(to_remove)
    return G


def p_original(fm, q_id, pqw, pwords, dls, pj, res, k1, k2, b, avdl, N, Nd, alpha, beta, gamma):
    """
    Fa il retrieve per la singola query
    :param fm: frequency matrix
    :param q_id: query id
    :param pqw: lista di query words per questa query
    :param pwords: dizionario delle parole
    :param dls: lunghezze dei documenti
    :param pj: indice dove scrivere in res
    :param res: matrice di output
    :param k1: param per bm25
    :param k2: param per bm25
    :param b: param per bm25
    :param avdl: lunghezza media dei documenti
    :param N: numero dei documenti
    :return: niente, salva la roba su res
    """
    # ignorare questa parte, fate finta che funzioni, alla fine avete il risultato di bm25
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

    idss_indx = np.argsort(ress)[::-1]

    idss = idss[idss_indx]
    ress = ress[idss_indx]

    idss_N = idss[0:Nd]
    ress_N = ress[0:Nd]

    G = get_graph_N(idss_N)

    try:
        auths, hubs = nx.hits(G)
    except nx.exception.NetworkXError, e:
        auths = {str(nid): 1.0 for nid in idss_N}
        hubs = {str(nid): 1.0 for nid in idss_N}
        print "HITS failed to converge"

    tmp_keys = auths.keys()
    for k in tmp_keys:
        if int(k) not in idss_N:
            auths.pop(k)
            hubs.pop(k)

    max_a, min_a = max(auths.values()), min(auths.values())
    max_h, min_h = max(hubs.values()), min(hubs.values())
    max_s, min_s = max(ress_N), min(ress_N)

    # normalizziamo
    for k, v in auths.iteritems():
        auths[k] = (auths[k]-min_a)/((max_a-min_a) if (max_a-min_a) > 0 else 1.0)
        hubs[k] = (hubs[k]-min_h)/((max_h-min_h) if (max_h-min_h) > 0 else 1.0)

    ress_N_tmp = (ress_N-min_s)/(max_s-min_s)
    ress_dict = {}
    for i, ind in enumerate(idss_N):
        ress_dict[str(ind)] = ress_N_tmp[i]

    hits_scores = dict()

    for k, v in auths.iteritems():
        hits_scores[k] = alpha*ress_dict[k] + beta * auths[k] + gamma * hubs[k]

    keys, values = np.array(hits_scores.keys()), np.array(hits_scores.values())

    indices = np.argsort(np.array(values))[::-1]
    keys = keys[indices].astype(int)

    idss[0:Nd] = keys

    res[pj, :, :] = np.vstack((idss, ress)).T


def retrieve(k1, b, k2, Nd, alpha, beta, gamma):
    """
    Questo fa tutto il lavoro di retrieval e salva il file results.txt
    :return:
    """
    # usa indexing per avere le strutture utili
    freq_mat, docs_length, words = indexing()

    query_stem = np.loadtxt("../data/query-stem.txt", dtype='str', delimiter="\t ")

    queries = {}

    # costuisco un dizionario che mappa gli id delle query => lista di query words
    for i, row in enumerate(query_stem):
        q_id = row[0]
        w = row[1]
        if q_id not in queries:
            queries[q_id] = [w]
        else:
            queries[q_id].append(w)
    # sara fatto cosi circa
    # {'1': [algebra, theorem], '2': [computer, system]......}

    # parametri per bm25
    N = freq_mat.shape[0]

    avdl = np.mean(docs_length)

    # salvo i risultati in questa matrice
    results = np.memmap("tmp", shape=(len(queries.keys()), N, 2), mode='w+', dtype='float')

    # per ogni query, chiamo p_original che fa il retrieval
    for pj, (q, lst) in enumerate(queries.iteritems()):
        p_original(freq_mat, q, lst, words, docs_length, pj, results, k1, k2, b, avdl, N, Nd, alpha, beta, gamma)

    # considera e retrieva solo i documenti con uno score almeno threshold
    # aumento threshold, diminuisce percentuale di documenti rilevanti retrieved
    threshold = 0.0

    f = open('results.txt', 'w')
    for j, query in enumerate(results):
        stuff_toprint = results[j, 0:1000]
        stuff_toprint = stuff_toprint[stuff_toprint[:, 1] > threshold]
        for i, row in enumerate(stuff_toprint):
            f.write("%s Q0 %s %s %s G12R9HITS\n" % (queries.keys()[j], int(row[0]+1), i+1, row[1]))
    f.close()

if __name__ == "__main__":
    k1 = float(sys.argv[1])
    b = float(sys.argv[2])
    k2 = float(sys.argv[3])
    Nd = float(sys.argv[4])
    alpha = float(sys.argv[5])
    beta = float(sys.argv[6])
    gamma = float(sys.argv[7])
    retrieve(k1, b, k2, Nd, alpha, beta, gamma)
