 
import sys
import lucene
from xml.dom import minidom
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer

from java.io import File
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer, ClassicAnalyzer
from org.apache.lucene.document import Document, Field
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.util import Version

from java.io import File
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.index import IndexReader
from org.apache.lucene.queryparser.classic import QueryParser, MultiFieldQueryParser, QueryParserBase
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.util import Version

'''
def index(indexdir):

  lucene.initVM()
  indexDir = SimpleFSDirectory(File(indexdir))
  writerConfig = IndexWriterConfig(Version.LUCENE_4_10_1, StandardAnalyzer())
  writer = IndexWriter(indexDir, writerConfig)

  print "%d docs in index" % writer.numDocs()
  print "Reading lines from sys.stdin..."
  for n, l in enumerate(sys.stdin):
    doc = Document()
    doc.add(Field("text", l, Field.Store.YES, Field.Index.ANALYZED))
    writer.addDocument(doc)
  print "Indexed %d lines from stdin (%d docs in index)" % (n, writer.numDocs())
  print "Closing index of %d docs..." % writer.numDocs()
  writer.close()
'''

def index(indexdir):
  lucene.initVM()
  indexDir = SimpleFSDirectory(File(indexdir))
  writerConfig = IndexWriterConfig(Version.LUCENE_4_10_1, EnglishAnalyzer())
  writer = IndexWriter(indexDir, writerConfig)

  f = open('data/docid.documento-xml.txt')
  st = PorterStemmer()
  for i, line in enumerate(f.readlines()):
    id, xmltext = line.split('\t')
    xmltext = xmltext.rstrip('\n')
    xmldoc = minidom.parseString(xmltext)
    title = xmldoc.getElementsByTagName("TITLE")
    title = "" if len(title) == 0 else title[0].childNodes[0].nodeValue
    authors = xmldoc.getElementsByTagName("AUTHORS")
    authors = "" if len(authors) == 0 else authors[0].childNodes[0].nodeValue
    abstract = xmldoc.getElementsByTagName("ABSTRACT")
    abstract = "" if len(abstract) == 0 else abstract[0].childNodes[0].nodeValue
    doc = Document()
    doc.add(Field("title", title, Field.Store.YES, Field.Index.ANALYZED))
    doc.add(Field("authors", authors, Field.Store.YES, Field.Index.ANALYZED))
    doc.add(Field("abstract", abstract, Field.Store.YES, Field.Index.ANALYZED))
    doc.add(Field("id", id, Field.Store.YES, Field.Index.NOT_ANALYZED))
    writer.addDocument(doc)
    print "indexed %s docs" % (i+1)

  writer.close()



def retrieve(indexdir, queries):
    lucene.initVM()
    f = open("results_lucene.txt", "w")
    analyzer = StandardAnalyzer(Version.LUCENE_4_10_1)
    reader = IndexReader.open(SimpleFSDirectory(File(indexdir)))
    searcher = IndexSearcher(reader)

    fields = ["title", "abstract", "authors"]

    st = PorterStemmer()
    for id, q in queries.iteritems():
        query = q
        tokenizer = RegexpTokenizer(r'\w+')
        qwords = tokenizer.tokenize(query)
        qwords_k = [st.stem(q) for q in qwords]
        query = " ".join(qwords_k)
        parser = MultiFieldQueryParser(Version.LUCENE_CURRENT, fields, analyzer)
        parser.setDefaultOperator(QueryParserBase.OR_OPERATOR)
        query = MultiFieldQueryParser.parse(parser, query)
        MAX = 1000
        hits = searcher.search(query, MAX)
        # print "Found %d document(s) that matched query '%s':" % (hits.totalHits, query)
        for i, hit in enumerate(hits.scoreDocs):
            f.write("%s Q0 %s %s %s G17R3\n" % (id, hit.doc+1, i+1, hit.score))
            # print hit.doc+1, hit.score
            # doc = searcher.doc(hit.doc)
            # print doc.get("authors").encode("utf-8")
    f.close()

def get_queries(queryfile):
    xmld = minidom.parse(queryfile)
    queries = {}
    qs = xmld.getElementsByTagName("DOC")
    for d in qs:
        id = d.childNodes[0].childNodes[0].nodeValue
        query = d.childNodes[1].nodeValue
        queries[id] = query
    return queries


if __name__ == "__main__":
    ind = "myindex/"
    # index(ind)
    # retrieve(ind)
    qs = get_queries('query-originale.txt')
    retrieve(ind, qs)