__author__ = 'giulio'

import numpy as np
import sys
import networkx as nx
# from joblib import cpu_count, Parallel, delayed


def get_matrix(row, col, file_name, dictionary):
    terms_mat = np.zeros(shape=(row, col))

    freq_docid_words = np.loadtxt(file_name, dtype='str')
    freq_docid_words[:, 1] = (freq_docid_words[:, 1].astype(int) - 1).astype(str)

    for r in freq_docid_words:
        freq, doc_id, word = int(r[0]), int(r[1]), r[2]
        word_index = dictionary[word]
        terms_mat[doc_id, word_index] = freq

    return terms_mat

def cossim(s1, s2):
    """
    Computes the cosine similarity of two vectors of doubles
    Returns the similarity,
    """
    # print s1, s2
    return 1.0 if np.array_equal(s1, s2) else 1 - np.degrees(
        np.arccos(np.dot(s1, s2) / (np.linalg.norm(s1) * np.linalg.norm(s2)))) / 180


def p_original(fm, q_id, pqw, pwords, dls, pj, res, k1, k2, b, avdl, N):
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

    N = 20

    idss_indx = np.argsort(ress)[::-1]

    idss = idss[idss_indx]
    ress = ress[idss_indx]

    idss_N = idss[0:N]
    ress_N = ress[0:N]

    occ_matrix = get_matrix(fm.shape[0], fm.shape[1], "../data/freq.docid.stem.txt", pwords)

    occ_matrix = occ_matrix[idss_N, :]

    queryVector = np.zeros(fm.shape[1])

    for qw in pqw:
        if qw in pwords:
            queryVector[pwords[qw]] = 1

    indxs = ~np.all(occ_matrix==0, axis=0)

    queryVector = queryVector[indxs]

    # rimuove colonne che hanno solo zeri
    occ_matrix = occ_matrix[:, indxs]

    u, s, vt = np.linalg.svd(occ_matrix.T, full_matrices=False)
    m = 10
    u[:, m:] = 0
    # u = u[:, :m]
    # sigma = np.diag(s)[:m, :m]
    sigma = np.diag(s)
    sigma_inv = np.linalg.inv(sigma)
    sigma_inv[:, m:] = 0
    sigma[:, m:] = 0
    # sigma[:, m:] = 0
    # vt = vt[:m, :]
    vt[m:, :] = 0

    # lowRankDocumentTermMatrix = np.dot(u, np.dot(sigma, vt))

    lowDimensionalQuery = np.dot(sigma_inv, np.dot(u.T, queryVector))

    similarity_array = np.zeros(N)

    for i, row in enumerate(vt.T):
        similarity_array[i] = cossim(lowDimensionalQuery, row)

    sortd = np.argsort(similarity_array)[::-1]

    ress_N = ress_N[sortd]
    idss_N = idss_N[sortd]

    idss[0:N] = idss_N
    # ress[0:N] = ress_N

    res[pj, :, :] = np.vstack((idss, ress)).T


def indexing():
    """
    Prende il file di stem, e ritorna:
    1. matrice delle occorrenze
    2. array che contiene le lunghezze dei docs
    3. il dizionario di parole
    :return:
    """
    # carica il file di stem in una matrice
    freq_docid_words = np.loadtxt("../data/freq.docid.stem.txt", dtype='str')

    # prende gli id e ci sottrae 1 cosi iniziano da zero
    freq_docid_words[:, 1] = (freq_docid_words[:, 1].astype(int) - 1).astype(str)

    # ritorna le parole uniche nel file di stem (rimuove i duplicati)
    words = np.unique(freq_docid_words[:,2])
    words_dict = {}
    for i, w in enumerate(words):
        # mettiamo nel nostro dizionario le parole e un indice che poi verra usato sulle colonne della matrice di freq,
        # la prima parola prendera' indice 0, la seconda 1, la terza 2 etc
        words_dict[w] = i

    n_words = len(words)
    print "%s distinct words" % n_words

    f = open('../data/docid.only-keywords.txt')
    # prendiamo il numero di documenti qui
    n_docs = len(f.readlines())
    print "%s documents" % n_docs

    f.close()

    # inizializzamo la matrice
    terms_mat = np.zeros(shape=(n_docs, n_words))

    # per ogni riga del file di stem, riempiamo le celle della matrice corrispondenti
    for r in freq_docid_words:
        freq, doc_id, word = int(r[0]), int(r[1]), r[2]
        # questo trova l'indice della parola nel nostro array di parole words
        # se word='algebra' si trova in posizione 5 dell'array di words, ritorna 5
        word_index = np.where(words == word)[0]
        # mettiamo in riga doc_id, colonna word_index (5 nell esempio qui sopra) l'occorrenza
        terms_mat[doc_id, word_index] = freq

    terms_mat = terms_mat.astype(float)

    # queste sono le lunghezze dei doc
    docs_length = terms_mat.sum(axis=1)

    # questa e' la matrice colle frequenze (dividiamo ogni colonna per la somma della colonna)
    # se un documento aveva la colonna [2, 4, 6,] ora e' diventata [1/6, 1/3, 1/2] (sono probabilita)
    terms_mat /= terms_mat.sum(axis=1)[:, None]

    # ritorna le 3 robe
    return terms_mat, docs_length, words_dict


def retrieve():
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
    k1 = 0.01
    k2 = 1.2
    b = 0.0
    avdl = np.mean(docs_length)

    # salvo i risultati in questa matrice
    results = np.memmap("tmp", shape=(len(queries.keys()), N, 2), mode='w+', dtype='float')

    # per ogni query, chiamo p_original che fa il retrieval
    for pj, (q, lst) in enumerate(queries.iteritems()):
        print q
        p_original(freq_mat, q, lst, words, docs_length, pj, results, k1, k2, b, avdl, N)

    # considera e retrieva solo i documenti con uno score almeno threshold
    # aumento threshold, diminuisce percentuale di documenti rilevanti retrieved
    threshold = 0.0

    f = open('results.txt', 'w')
    for j, query in enumerate(results):
        # indx = np.argsort(results[j, :, 1])[::-1][0:1000]

        stuff_toprint = results[j, 0:1000]
        stuff_toprint = stuff_toprint[stuff_toprint[:, 1] > threshold]
        for i, row in enumerate(stuff_toprint):
            f.write("%s Q0 %s %s %s G12R9PR\n" % (queries.keys()[j], int(row[0]+1), i+1, row[1]))
    f.close()

if __name__ == "__main__":
    # invoca retrieve
    retrieve()
