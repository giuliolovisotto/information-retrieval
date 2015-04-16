__author__ = 'giulio'

import numpy as np

# ricorda che gli stem che ti ha fornito non rappresentano bisogno informativi
# esempio cercare l'autore (query 2) dire 'non sono interested in ___' (query 6) etc


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

    query_stem = np.loadtxt("query-stem.txt", dtype='str', delimiter="\t ")

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
    k1 = 1.2
    k2 = 1.2
    b = 0.5
    avdl = np.mean(docs_length)

    f = open('results.txt', 'w')

    for q in queries.keys():
        results = np.zeros(shape=(N, 2))
        print "Query %s" % q
        query_words = queries[q]
        print query_words
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
                    results[i, :] = np.array([i, score])

        indx = np.argsort(results[:, 1])[::-1][0:1000]
        stuff_toprint = results[indx]
        stuff_toprint = stuff_toprint[stuff_toprint[:, 1] > 0]
        for i, row in enumerate(stuff_toprint):
            f.write("%s Q0 %s %s %s G17R3\n" % (q, int(row[0]), i+1, row[1]))

        f.close()
        break


            #if score > 0:
                #print "doc %s, score %s" % (i+1, score)
    #indx = np.argsort(results[:, 1])[::-1]
    #print results[indx]



if __name__ == "__main__":
    print "ciao"
    retrieve()