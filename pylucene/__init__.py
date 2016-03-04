'''
Created on Jun 12, 2014

@author: luamct
'''

import sys
import lucene
from config import INDEX_PATH, DATA

from java.io import File
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.index import FieldInfo, IndexWriter, IndexWriterConfig, DirectoryReader, Term, MultiFields
from org.apache.lucene.store import RAMDirectory, SimpleFSDirectory
from org.apache.lucene.util import Version
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.search.similarities import BM25Similarity
from org.apache.lucene.queryparser.classic import QueryParser, MultiFieldQueryParser
from org.apache.lucene.queries import TermsFilter

# from lucene import RamDirectory, System, File, Document, \
# 									Field, StandardAnalyzer, IndexWriter, Version

class DocField() :
	def __init__(self, name, **kwargs):
		self.name = name
		self.props = kwargs


class Index :
	
	def __init__(self, folder=None, fields=[], similarity="tfidf"):

		self.jcc = lucene.initVM()

		if folder :
			self.directory = SimpleFSDirectory(File(folder))
		else:
			self.directory = RAMDirectory()

		self.fields = {}

		for field in fields :
			ft = FieldType()
			for pname, pvalue in field.props.items() :
				setter = getattr(ft, "set"+pname.capitalize())
				setter(pvalue)

			ft.setIndexOptions(FieldInfo.IndexOptions.DOCS_AND_FREQS)
# 			ft.setOmitNorms(True)

			self.fields[field.name] = ft

		self.similarity = similarity.lower()
		self.analyzer = StandardAnalyzer(Version.LUCENE_CURRENT)
		self.writer = None
		self.searcher = None


	def attach_thread(self) :
		self.jcc.attachCurrentThread()


	def open_writer(self) :

		config = IndexWriterConfig(Version.LUCENE_CURRENT, self.analyzer)
		config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)

		self.writer = IndexWriter(self.directory, config)


	def add(self, **doc) :

		if not self.writer :
			self.open_writer()

		d = Document()
		for field, value in doc.items() :
#			try :
				d.add(Field(field, value, self.fields[field]))
#			except Exception, e :
#				print 
#				print "Fudeu"
#				pass

		self.writer.addDocument(d)


	def commit(self) :
		self.writer.commit()


	def close(self):
		if self.writer :
			self.writer.close()


	def open_searcher(self):
		self.reader = DirectoryReader.open(self.directory)
		self.searcher = IndexSearcher(self.reader)
		if (self.similarity == "bm25") :
			self.searcher.setSimilarity(BM25Similarity())


	def preprocess_query(self, query, fields, mode="ANY"):
		'''
		Fix query according to provided mode. If the value is not supported, 
		the query remains unchanged
		'''

		terms = query.lower().strip().split()
		if mode=="ANY" :
			query = " OR ".join(terms)
		elif mode=="ALL":
			query = " AND ".join(terms)
		else :
			print "Invalid mode parameter '%s'." % mode
			
		query = QueryParser.escape(query)
		parser = MultiFieldQueryParser(Version.LUCENE_CURRENT, fields, self.analyzer)
		query = MultiFieldQueryParser.parse(parser, query)
		return query

			
	def search(self, query, 
						 search_fields, 
						 return_fields, 
						 filter=None, 
						 ignore=set(),
						 mode="ANY",
						 return_scores=False,
						 limit=1000000):
		'''
		Search documents in the index using a standard analyzer (tokenizes and 
		removes top words). Supports two search modes: ANY and ALL
		  ANY: include documents that contain at least one term of the query.
		  ALL: include only documents that contain all terms of the query. 
		'''

		if not self.searcher :
			self.open_searcher()

		# Return empty results if query is empty (Lucene can't handle it nicely)
		if query.strip()=='':
			if return_scores :
				return [], []
			else:
				return []

		
		query = self.preprocess_query(query, search_fields, mode)

		# If limit is not provided, return all matched documents. A little hack is required
		# to do that. We query for one document and get the count total matched documents. 
#		if not limit :
#			hits = self.searcher.search(query, 1)
#			limit = hits.totalHits

		# Fetch more than asked in case we have to remove entries from the ignore set
		if limit!=None:
			limit += len(ignore)

		hits = self.searcher.search(query, filter, limit)
		hits = hits.scoreDocs

		docs = []
		for hit in hits :
			doc = self.searcher.doc(hit.doc)
			if doc['id'] not in ignore:
				docs.append([doc[f] for f in return_fields])

		if return_scores :
			scores = [hit.score for hit in hits]
			return docs[:limit], scores[:limit]

		return docs[:limit]


	def explain(self, query, fields, doc):

		if not self.searcher :
			self.open_searcher()

		query = QueryParser.escape(query)

		parser = MultiFieldQueryParser(Version.LUCENE_CURRENT, fields, self.analyzer)
		query = MultiFieldQueryParser.parse(parser, query)
		
		return self.searcher.explain(query, doc)


	def get_documents(self, doc_ids, fields) :
		
		docs = []
		for doc_id in doc_ids:
			doc = self.reader.document(doc_id)
			if isinstance(fields, basestring) :
				docs.append(doc.get(fields))
			else :
				docs.append( {f:doc.get(f) for f in fields} )

		return docs

	
	def get_query_scores(self, query, fields, doc_ids, mode="ANY") :

		# Creates pre-filter to ignore all other documents
		filter = TermsFilter([Term("id", id) for id in doc_ids])

		query = self.preprocess_query(query, fields, mode)
		hits = self.searcher.search(query, filter, len(doc_ids)).scoreDocs

		# Creates scores' mapping using entity id instead of internal index id
		scores = {str(self.reader.document(hit.doc).get("id")): hit.score for hit in hits}

		# Normalize to 0..1 interval
#		n = 1.0/sum(scores.values())
#		scores

		# Adds to the mapping entries for the non-returned docs (no term found)
		for doc_id in doc_ids:
			if doc_id not in scores :
				scores[doc_id] = 0.0
				
		return scores

		

if __name__ == "__main__" :

	index = Index(DATA + "index_csx_notext")

	filter = TermsFilter([Term("id", "10.1.1.130.1300"), 
												Term("id", "10.1.1.3.4530")])
#	filter = None
	docs = index.search("image descriptors", 
											 search_fields=["title", "abstract"], 
											 return_fields=["id", "title"], 
											 filter=filter, limit=10)

#	titles = index.get_documents(ids, ["id", "title"])
	print "\n".join(map(str, docs))
	sys.exit()

	fields = [DocField("id", stored=True, indexed=True),
						DocField("text", stored=True, indexed=True)]
	index = Index(fields=fields)
	texts = ["just writing ", "what ever dude", "el dudino", "your dude", "the Dude"]

	for i, text in enumerate(texts) :
		index.add(id='doc_%d'%(i+1), text=text)
	index.commit()

	ids, scores = index.search("dude+ever", ["text"], limit=10)
	print index.get_documents(ids, "id")

	# Try out some filters
	filter = TermsFilter([Term("id", "doc_2")])
	ids, scores = index.search("dude+ever", ["text"], filter, limit=10)
	print index.get_documents(ids, "id")

	fields = MultiFields.getMergedFieldInfos(index.reader).iterator()
	for f in fields:
		print f.attributes()
#	print filter.getDocIdSet(index.reader)
	
	
	
	

