__author__ = 'giulio'

import numpy as np
import sys
from joblib import cpu_count, Parallel, delayed
# ricorda che gli stem che ti ha fornito non rappresentano bisogno informativi
# esempio cercare l'autore (query 2) dire 'non sono interested in ___' (query 6) etc

def p(fm, pqw, pwords, dls, pj, res, k1, k2, b, avdl, N):
    print pj
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
    # ress = np.multiply(ress, tf2s)
    ress = ress.sum(axis=1)
    idss = np.arange(0, red_fm.shape[0])
    res[pj, :, :] = np.vstack((idss, ress)).T
    '''
    for i, row in enumerate(fm):
        score = 0
        # print "docuemnto %s" % i
        dl = dls[i]
        for qw in actual_qw:
            #print qw, i

            index_of_qw = pwords[qw]
            n_i = np.count_nonzero(fm[:, index_of_qw])
            #print "indice of queryword: %s, # doc compare: %s" % (index_of_qw, n_i)
            # num = N-n_i+0.5
            # den = n_i + 0.5
            # print num, den
            idf = np.log((N - n_i + 0.5)/(n_i + 0.5))
            #print "idf: %s" % idf
            #print "doc length %s" % dl
            K = k1*((1-b) + b*(dl/avdl))
            #print "K: %s, k1:%s, b:%s, avdl:%s " % (K, k1, b, avdl)
            #print "valore di %s in doc %s: %s" % (qw, i, fm[i, index_of_qw])
            tf1 = (k1 + 1)*fm[i, index_of_qw]/(K + fm[i, index_of_qw])
            tf2 = (k2 + 1)*1/(k2 + 1)
            #print tf1, tf2, idf
            score += idf * tf1 * tf2
        # print "score per doc %s: %s" %(i, score)
        if score != 0:
            res[pj, i, :] = np.array([i, score])
            # print "salvo in %s,%s la roba %s,%s" % (pj, i, i, score)
    '''

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
    # n_docs = 5
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
    print queries
    N = freq_mat.shape[0]
    k1 = 0.01
    k2 = 1.2
    b = 0.0  # lascio basso perche normalizzare sulla lunghezza e' un po inutile nel nostro caso (abbiamo solo le kw)
    avdl = np.mean(docs_length)

    results = np.memmap("tmp", shape=(len(queries.keys()), N, 2), mode='w+', dtype='float')
    results[:] = 0
    Parallel(n_jobs=cpu_count())(delayed(p)(
        freq_mat, lst, words, docs_length, pj, results, k1, k2, b, avdl, N
    ) for pj, (q, lst) in enumerate(queries.iteritems()))

    '''
    for j, q in enumerate(queries.keys()):

        sys.stdout.write("\rQuery %s of %s" % (j+1, len(queries)))
        # print "Query %s" % q
        query_words = queries[q]
        # print query_words
        for i, row in enumerate(freq_mat):
            score = 0
            for qw in query_words:
                if qw not in words:
                    score += 0
                else:
                    index_of_qw = words[qw]
                    n_i = np.count_nonzero(freq_mat[:, index_of_qw])
                    idf = np.log((N - n_i + 0.5)/(n_i + 0.5))
                    dl = docs_length[i]
                    K = k1*((1-b) + b*(dl/avdl))

                    tf1 = (k1 + 1)*freq_mat[i, index_of_qw]/(K + freq_mat[i, index_of_qw])
                    tf2 = (k2 + 1)*1/(k2 + 1)
                    score += idf * tf1 * tf2
                    results[j, i, :] = np.array([i, score])

        # indx = np.argsort(results[j, :, 1])[::-1][0:1000]
        # stuff_toprint = results[j, indx]
        # stuff_toprint = stuff_toprint[stuff_toprint[:, 1] > 0]
        # for i, row in enumerate(stuff_toprint):
        #     f.write("%s Q0 %s %s %s G17R3\n" % (q, int(row[0]), i+1, row[1]))
        break
    # f.close()
    '''

    # considera e retrieva solo i documenti con uno score almeno threshold
    # aumento threshold, diminuisce percentuale di documenti rilevanti retrieved
    threshold = 0.0

    f = open('results.txt', 'w')
    for j, query in enumerate(results):
        indx = np.argsort(results[j, :, 1])[::-1][0:100]
        stuff_toprint = results[j, indx]
        stuff_toprint = stuff_toprint[stuff_toprint[:, 1] > threshold]
        for i, row in enumerate(stuff_toprint):
            f.write("%s Q0 %s %s %s G12R9\n" % (queries.keys()[j], int(row[0]+1), i+1, row[1]))
    f.close()



            #if score > 0:
                #print "doc %s, score %s" % (i+1, score)
    #indx = np.argsort(results[:, 1])[::-1]
    #print results[indx]



if __name__ == "__main__":

    retrieve()