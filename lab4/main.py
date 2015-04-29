__author__ = 'giulio'

import numpy as np
import sys
from joblib import cpu_count, Parallel, delayed


def get_R(query_id):
    a = np.loadtxt("data/qrels-treceval.txt", dtype='str')[:, 0].astype(int)
    R = np.count_nonzero(a == int(query_id))
    return R


def get_r_i_pseudo(mat_freq, rel_ids, query_term):
    mr = np.copy(mat_freq[rel_ids, :])
    r_i = np.count_nonzero(mr[:, int(query_term)])
    return r_i


def get_r_i(mat_freq, query_id, query_term):
    a = np.loadtxt("data/qrels-treceval.txt", dtype='str')
    b = a[:, 0].astype(int)
    a = a[b == int(query_id)]
    a = a[:, 2].astype(int)
    a -= 1
    mr = np.copy(mat_freq[a, :])
    r_i = np.count_nonzero(mr[:, int(query_term)])
    return r_i


def p_pseudo(fm, q_id, pqw, pwords, dls, pj, res, k1, k2, b, avdl, N):
    print pj
    actual_qw = []
    for qw in pqw:
        if qw in pwords:
            actual_qw.append(qw)

    for i, row in enumerate(fm):
        score = 0
        for jj, qw in enumerate(actual_qw):
            index_of_qw = pwords[qw]
            n_i = np.count_nonzero(fm[:, index_of_qw])
            idf = np.log((N - n_i + 0.5) * 0.5/(n_i + 0.5) * 0.5)
            dl = dls[i]
            K = k1*((1-b) + b*(dl/avdl))
            tf1 = (k1 + 1)*fm[i, index_of_qw]/(K + fm[i, index_of_qw])
            tf2 = (k2 + 1)*1/(k2 + 1)
            score += idf * tf1 * tf2
            res[pj, i, :] = np.array([i, score])

    ranking = res[pj, :, :]
    R = 10
    indx = np.argsort(ranking[:, 1])[::-1][0:R]
    relevants = ranking[indx, :]
    rel_ids = relevants[:, 0].astype(int)

    r_is = []
    for qw in actual_qw:
        index_of_qw = pwords[qw]
        r_is.append(get_r_i_pseudo(fm, rel_ids, index_of_qw))

    r_is = np.array(r_is)

    for i, row in enumerate(fm):
        score = 0
        for jj, qw in enumerate(actual_qw):
            index_of_qw = pwords[qw]
            r_i = r_is[jj]
            n_i = np.count_nonzero(fm[:, index_of_qw])
            idf = np.log((N - n_i - R + r_i + 0.5)*(r_i + 0.5)/(n_i - r_i + 0.5)*(R - r_i + 0.5))
            dl = dls[i]
            K = k1*((1-b) + b*(dl/avdl))
            tf1 = (k1 + 1)*fm[i, index_of_qw]/(K + fm[i, index_of_qw])
            tf2 = (k2 + 1)*1/(k2 + 1)
            score += idf * tf1 * tf2
            res[pj, i, :] = np.array([i, score])


def p_esplicito(fm, q_id, pqw, pwords, dls, pj, res, k1, k2, b, avdl, N):
    print pj
    actual_qw = []
    r_is = []
    for qw in pqw:
        if qw in pwords:
            actual_qw.append(qw)
            index_of_qw = pwords[qw]
            r_is.append(get_r_i(fm, q_id, index_of_qw))

    R = get_R(q_id)

    r_is = np.array(r_is)

    for i, row in enumerate(fm):
        score = 0
        for jj, qw in enumerate(actual_qw):
            index_of_qw = pwords[qw]
            # print q_id, index_of_qw
            r_i = r_is[jj]
            # print "termine: %s, compare in %s" % (qw, r_i)
            n_i = np.count_nonzero(fm[:, index_of_qw])
            idf = np.log((N - n_i - R + r_i + 0.5)*(r_i + 0.5)/(n_i - r_i + 0.5)*(R - r_i + 0.5))
            dl = dls[i]
            K = k1*((1-b) + b*(dl/avdl))

            tf1 = (k1 + 1)*fm[i, index_of_qw]/(K + fm[i, index_of_qw])
            tf2 = (k2 + 1)*1/(k2 + 1)
            score += idf * tf1 * tf2
            res[pj, i, :] = np.array([i, score])


def indexing():
    freq_docid_words = np.loadtxt("data/freq.docid.stem.txt", dtype='str')

    freq_docid_words[:, 1] = (freq_docid_words[:, 1].astype(int) - 1).astype(str)

    words = np.unique(freq_docid_words[:,2])
    words_dict = {}
    for i, w in enumerate(words):
        words_dict[w] = i

    n_words = len(words)
    print "%s distinct words" % n_words

    f = open('data/docid.only-keywords.txt')

    n_docs = len(f.readlines())
    print "%s documents" % n_docs

    f.close()

    terms_mat = np.zeros(shape=(n_docs, n_words))

    for r in freq_docid_words:
        freq, doc_id, word = int(r[0]), int(r[1]), r[2]
        word_index = np.where(words==word)[0]
        terms_mat[doc_id, word_index] = freq

    terms_mat = terms_mat.astype(float)

    docs_length = terms_mat.sum(axis=1)

    terms_mat /= terms_mat.sum(axis=1)[:, None]

    return terms_mat, docs_length, words_dict


def retrieve():
    freq_mat, docs_length, words = indexing()

    query_stem = np.loadtxt("data/query-stem.txt", dtype='str', delimiter="\t ")

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
    k1 = 0.01
    k2 = 1.2
    b = 0.0  # lascio basso perche normalizzare sulla lunghezza e' un po inutile nel nostro caso (abbiamo solo le kw)
    avdl = np.mean(docs_length)

    results = np.memmap("tmp", shape=(len(queries.keys()), N, 2), mode='w+', dtype='float')

    #for pj, (q, lst) in enumerate(queries.iteritems()):
    #    p_pseudo(freq_mat, q, lst, words, docs_length, pj, results, k1, k2, b, avdl, N)

    Parallel(n_jobs=cpu_count())(delayed(p_pseudo)(
        freq_mat, q, lst, words, docs_length, pj, results, k1, k2, b, avdl, N
    ) for pj, (q, lst) in enumerate(queries.iteritems()))

    # considera e retrieva solo i documenti con uno score almeno threshold
    # aumento threshold, diminuisce percentuale di documenti rilevanti retrieved
    threshold = 0.0

    f = open('results.txt', 'w')
    for j, query in enumerate(results):
        indx = np.argsort(results[j, :, 1])[::-1][0:1000]
        stuff_toprint = results[j, indx]
        stuff_toprint = stuff_toprint[stuff_toprint[:, 1] > threshold]
        for i, row in enumerate(stuff_toprint):
            f.write("%s Q0 %s %s %s G12R9\n" % (queries.keys()[j], int(row[0]+1), i+1, row[1]))
    f.close()

if __name__ == "__main__":

    retrieve()