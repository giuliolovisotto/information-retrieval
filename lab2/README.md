# Lab 2 

We need to extract _tfidf_ weights given the collection, and produce in output a file formatted like this:  

word1 doc\_id weight  
word2 doc\_id weight  
word3 doc\_id weight  
...

We already have:
* stemmed words frequency
* a stoplist
* the keywords for every document
Stopwords have already been removed from the keywords so we can ignore _stopping_.  
Since keywords are already stemmed we can ignore _stemming_ too.  

The script *weights.py** builds a dictionary with the unique words, then builds a matrix of n_docs x n_words with occurencies using the frequency file:  

doc_id | word1 | word2 ...
--------------------------
doc1   | 5     | 0
doc2   | 1     | 2
doc3 ..| ..
...

with this matrix we use the _TfidfTransformer_ of the [scikit-learn](www.scikit-learn.org) module to get a matrix with the tfidf weights. Finally we save them to the file.