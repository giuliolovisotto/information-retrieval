__author__ = 'giulio'

import os
import numpy as np
from scipy import sparse, io
import json


def indexing():
    """
    Prende il file di stem, e ritorna:
    1. matrice delle occorrenze
    2. array che contiene le lunghezze dei docs
    3. il dizionario di parole
    :return:
    """

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