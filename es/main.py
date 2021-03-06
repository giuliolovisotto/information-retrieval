__author__ = 'giulio'
import numpy as np
import sys
import os
import json
import time
from scipy import sparse, io
from joblib import cpu_count, Parallel, delayed
# ricorda che gli stem che ti ha fornito non rappresentano bisogno informativi
# esempio cercare l'autore (query 2) dire 'non sono interested in ___' (query 6) etc

def p(fm, q_id, pqw, pwords, dls, pj, res, k1, k2, b, avdl, N):
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
    res[pj, :, :] = np.vstack((idss, ress)).T


def indexing():
    if not os.path.isfile("terms_mat.mtx"):
        freq_docid_words = np.loadtxt("../data/freq.docid.stem.txt", dtype='str')

        freq_docid_words[:, 1] = (freq_docid_words[:, 1].astype(int) - 1).astype(str)

        words = np.unique(freq_docid_words[:,2])
        words_dict = {}
        for i, w in enumerate(words):
            words_dict[w] = i

        n_words = len(words)

        f = open('../data/docid.only-keywords.txt')

        n_docs = len(f.readlines())

        f.close()

        terms_mat = np.zeros(shape=(n_docs, n_words))

        for r in freq_docid_words:
            freq, doc_id, word = int(r[0]), int(r[1]), r[2]
            word_index = np.where(words==word)[0]
            terms_mat[doc_id, word_index] = freq

        terms_mat = terms_mat.astype(float)

        docs_length = terms_mat.sum(axis=1)

        terms_mat /= terms_mat.sum(axis=1)[:, None]

        terms_mat = sparse.csr_matrix(terms_mat)
        io.mmwrite("terms_mat.mtx", terms_mat)
        np.savetxt("docs_length.txt", docs_length)
        json.dump(words_dict, open("words_dict.txt", 'w'))

    else:
        terms_mat = np.array(io.mmread("terms_mat.mtx").todense())
        docs_length = np.loadtxt("docs_length.txt")
        words_dict = json.load(open("words_dict.txt"))

    return terms_mat, docs_length, words_dict


def retrieve(k, b):
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

    N = freq_mat.shape[0]
    k1 = k
    k2 = 1.2
    # b = 0.0  # lascio basso perche normalizzare sulla lunghezza e' un po inutile nel nostro caso (abbiamo solo le kw)
    avdl = np.mean(docs_length)

    results = np.memmap("tmp", shape=(len(queries.keys()), N, 2), mode='w+', dtype='float')
    results[:] = 0
    for pj, (q, lst) in enumerate(queries.iteritems()):
        p(freq_mat, q, lst, words, docs_length, pj, results, k1, k2, b, avdl, N)

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
    k = float(sys.argv[1])
    b = float(sys.argv[2])
    retrieve(k, b)

