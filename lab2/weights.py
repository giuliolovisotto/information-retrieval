import numpy as np
import sklearn.feature_extraction.text as ftextr

# matrix with frequencies, 3 columns
# this matrix has a row for every line in the file, in every row
# there are 3 elements (freq, doc_id, word)
freq_docid_words = np.loadtxt("data/freq.docid.stem.txt", dtype='str')

# update document ids so they start from 0 (they start from 1)
freq_docid_words[:, 1] = (freq_docid_words[:, 1].astype(int) - 1).astype(str)

# get unique words, this is our dictionary
words = np.unique(freq_docid_words[:,2])

# number of words in dictionary
n_words = len(words)
print "%s distinct words" % n_words

# open file just to count number of documents
f = open('data/docid.only-keywords.txt')

# get line count which is number of documents in the collection
n_docs = len(f.readlines())
print "%s documents" % n_docs

# close file
f.close()

# initialize occurencies matrix
# every row is a document
# every column is a term in the dictionary
# every value (i, j) will be the number of occurencies of term j in document i
terms_mat = np.zeros(shape=(n_docs, n_words))

# for every row in frequencies file (which is now in our matrix)
for r in freq_docid_words:
    # get the three fields
    freq, doc_id, word = int(r[0]), int(r[1]), r[2]
    # get the index of the word in the dictionary,
    # np.where returns an array with the indexes of where the condition is true
    # in the array. 
    # so if we have an array     words = ['a', 'b', 'c', 'd']
    # and we use                 np.where(words=='b')
    # the result will be         np.array([1])
    word_index = np.where(words==word)[0]
    # set the corresponding cell to the frequency
    terms_mat[doc_id, word_index] = freq

# initialize tfidf transformer
transf = ftextr.TfidfTransformer()

# compute tfidf
# the method fit_transform takes a matrix with occurencies
# and returns a matrix with the tfidfs
tfidf = transf.fit_transform(terms_mat)

print tfidf.shape

# now we need to save all the non zero elements in our output file with weights
# get non-zero elements indices
rows, cols = np.nonzero(tfidf)

# open file for write
outf = open('weights.csv', 'w')

# for every non-zero element
for row,col in zip(rows,cols):
    # use the column index to get the corresponding word
    word = words[col]
    # remember that we are using ids - 1 so fix that
    doc_id = row+1
    # write to file the
    # 1) word
    # 2) the docid
    # 3) the tfidf for the word in the doc
    # 4) the number of occurencies for word in the doc
    # we store the number of occurencies too because it is useful for retrieval models not directly
    # based on tfidf score
    outf.write("%s %s %s\n" % (word, doc_id, tfidf[row, col]))

# close
outf.close()

print "Written output to weights.csv"




