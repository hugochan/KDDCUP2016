'''
Created on Jul 7, 2014

@author: luamct
'''

import chardet
import numpy as np
import networkx as nx
from mymysql import MyMySQL
from collections import defaultdict
from exceptions import TypeError

import itertools
import logging as log
#import words
from config import DATA, DATASET
import config
import line_profiler
import csv
import time


#MAX_WORDS_PER_DOC = 5

# Database connection
db = MyMySQL(db=DATASET, user="root")


########################################
## Helper methods                     
########################################

def sorted_tuple(a,b):
	''' Simple pair sorting to avoid repetitions when inserting into set or dict. '''
	return (a,b) if a<b else (b,a)


def get_unicode(s):
	''' Return an unicode of the string encoded with the most likely format. '''
	if not s :
		return unicode("", "utf-8")

	return unicode(s, chardet.detect(s)['encoding'])


def get_all_edges() :
	''' Retrieve all edges from the database. '''
	return db.select(fields=["citing", "cited"], table="graph")

def show_stats(graph) :
	print "%d nodes and %d edges." % (graph.number_of_nodes(), graph.number_of_edges())
	
def write_graph(graph, outfile):
	'''
	Write the networkx graph into a file in the gexf format.
	'''
	log.info("Dumping graph: %d nodes and %d edges." % (graph.number_of_nodes(), graph.number_of_edges()))
	nx.write_gexf(graph, outfile, encoding="utf-8")

def get_paper_year(paper_id) :
	'''
	Returns the year of the given paper as stored in the DB.
	'''
	year = db.select_one(fields="year", table="papers", where="id='%s'"%paper_id)
	return (int(year) if year else 0)

def normalize_edges(edges) :
	'''
	Normalize the weight on given edges dividing by the maximum weight found.
	'''
	wmax = 0.0
	for _u,_v,w in edges :
		wmax = max(w, wmax)

	return [(u,v,w/float(wmax)) for u,v,w in edges]


def similarity(d1, d2):
	'''
	Cosine similarity between sparse vectors represented as dictionaries. 
	'''
	sim = 0.0
	for k in d1 :
		if k in d2 :
			sim += d1[k]*d2[k]

	dem = np.sqrt(np.square(d1.values()).sum()) * np.sqrt(np.square(d2.values()).sum())
	return sim/dem


def get_rules_by_lift(transactions, min_lift=1.0) :
	'''
	Get relevant association between topics assuming that every document is a 
	transaction and relevant topics for each documents are items in that transaction.
	We then use the lift metric to find strong co-occurrences between topics.
	'''

	freqs1 = defaultdict(int)  # Frequencies of 1-itemsets
	freqs2 = defaultdict(int)  # Frequencies of 2-itemsets
	for trans in transactions :
		for i in trans :
			freqs1[i] += 1

		# If there are at least 2 items, let's compute pairs support
		if len(trans) >= 2 :
			for i1, i2 in itertools.combinations(trans, 2) :
				freqs2[sorted_tuple(i1, i2)] += 1

	n = float(len(transactions))

	# Check every co-occurring ngram
	rules = []
	for (i1,i2),f in freqs2.items() :

		# Consider only the ones that appear more than once together,
		# otherwise get_rules_by_lift values can be huge and not really significant 
		if f > 1 :
			lift = f*n/(freqs1[i1]*freqs1[i2])

			# Include only values higher than min_lift
			if lift>=min_lift :
				rules.append( (i1, i2, np.log10(lift)) )

	return rules


def make_csv(file_path, headers, rows):
	'''
	Builds CSV file with given headers and rows.
	'''
	print "%s: %d rows." % (file_path, len(rows))
	with open(file_path, 'wb') as csvfile:
		w = csv.writer(csvfile)

		# Header
		w.writerow(headers)
		for row in rows:
			w.writerow(row)

########################################
## Class definitions 
########################################


class ModelBuilder :
	'''
	Main class for building the graphical model. The layers are built separately in their
	corresponding methods. Every layer is cached in a folder defined by the main parameters. 
	'''
	def __init__(self) :
		'''
		Initializes structures and load data into memory, such as the text index and 
		the citation graph. 
		'''
		# Maps original ids of entities from all layers to an universal 
		# incremental interger id for each node. There's actually two
		# levels of mapping, first for the node type and then for the actual id.
		# ids[node_type][orig_id] = unique_id
		self.pub_ids = {}
		self.author_ids = {}
		self.topic_ids = {}
		self.word_ids = {}
		self.next_id = 0

		# Create a helper boolean to check if citation contexts are 
		# going to be used (some datasets don't have it available) 
		self.use_contexts = False

		# Load vocabulary for the tokens in the citation contexts
#		if self.use_contexts :
#			self.ctxs_vocab, self.nctx = words.read_vocab(DATA + "contexts_tfidfs_tokens.txt")

		log.info("ModelBuilder constructed.")


	def get_next_id(self) :
		self.next_id += 1
		return self.next_id


	def get_papers_layer(self, limit=None) :
		'''
		Just get all publications (or a sample, if limit was provided) and
		all the citations between those publications.
		'''
#		ids = self.pub_ids

		pubs = set(map(str, db.select("id", table="papers", limit=limit)))
#		for pub_id in pubs :
#			ids[pub_id] = self.get_next_id()

		citations = db.select(["citing", "cited"], table="graph")
		citation_edges = []
		for (u,v) in citations:
			if (u in pubs) and (v in pubs) :
				citation_edges.append( (str(u), str(v), 1.0) ) 

		return list(pubs), citation_edges

	
	def get_authors(self, pub_id) :
		'''
		Return the authors associated with the given paper, if available. 
		'''
		return self.authorships[pub_id]
#		return db.select("author_id", table="authorships", where="paper_id='%s'" % doc_id)


	def get_coauthorship_edges(self, authors):
		'''
		Return all the collaboration edges between the given authors. 
		Edges to authors not provided are not included.
		'''
		# For efficient lookup	
		authors = set(authors)

		edges = set()
		for author_id in authors:
			coauthorships = db.select_query("""SELECT b.author_id FROM authorships a, authorships b 
																				 WHERE (a.author_id=%d) AND (b.author_id!=%d) AND a.paper_id=b.paper_id""" \
																				 % (author_id, author_id))

			# Count coauthored pubs
			coauthors = defaultdict(int)
			for (coauthor,) in coauthorships :
				if coauthor in authors : 
					coauthors[(author_id, coauthor)] += 1 

			for (a1, a2), npapers in coauthors.items() :

				# Apply log transformation to smooth values and avoid outliers 
				# crushing other values after normalization
				weight = 1.0 + np.log(npapers)

				if (a1 in authors) and (a2 in authors) :
					edge = (a1,a2,weight) if a1<a2 else (a2,a1,weight)
					edges.add(edge)

		# Normalize by max value and return them as a list
		return normalize_edges(edges)


	def get_authorship_edges(self, papers_authors) :
		'''
		Return authorship edges [(doc_id, author), ...]
		'''
		edges = []
		for doc_id, authors in papers_authors.items() :
			edges.extend( [(doc_id, author, 1.0) for author in authors] )

		return edges


	def get_authors_layer(self, papers) :
		'''
		Retrieve relevant authors from DB (author of at least one paper given as argument)
		and assemble co-authorship and authorship nodes and edges.
		'''
		# Load authorships to speed lookups
		auth_rows = db.select(["paper_id", "author_id"], table="authorships")

		authorships = defaultdict(list)
		for pub_id, author_id in auth_rows :
			authorships[str(pub_id)].append(author_id)

		# Load author names
		author_names = {id:name.strip() for id,name in db.select(["cluster", "name"], "authors_clean")}

		all_authors = set()
		papers_authors = {}
		for paperid in papers :

			paper_authors = authorships[paperid]

			papers_authors[paperid] = paper_authors
			all_authors.update(paper_authors)


		coauth_edges = self.get_coauthorship_edges(all_authors)
		auth_edges = self.get_authorship_edges(papers_authors)
#		all_authors = list(all_authors)

		authors = [(id, author_names[id].encode("UTF-8")) for id in all_authors]
		return authors, coauth_edges, auth_edges


#	def get_relevant_topics(self, doc_topics, ntop=None, above=None) :
#		'''
#		Get the most important topics for the given document by either:
#			* Taking the 'ntop' values if 'ntop' id provided or 
#			* Taking all topics with contributions greater than 'above'.
#		'''
#		if ntop :
#			return np.argsort(doc_topics)[::-1][:ntop]
#
#		if above :
#			return np.where(doc_topics>above)[0]
#
#		raise TypeError("Arguments 'ntop' and 'above' cannot be both None.")


	def get_topics_layer_db(self, doc_ids, ntopics) :
		'''
		Run topic modeling for the content on the given papers and assemble the topic nodes 
		and edges.
		'''
# 		topics, doc_topics, tokens = topic_modeling.get_topics_online(doc_ids, ntopics=200, beta=0.1, 
# 																																cache_folder=self.cache_folder, ign_cache=False)

		# Load topics' assignment to speed up topics layer lookups
		doc_topics_rows = db.select(["paper_id", "topic_id", "value"], 
																table="doc_topics", 
																where="value>=%f"%config.MIN_TOPIC_VALUE)

		doc_topics = defaultdict(list)
		for pub_id, topic_id, value in doc_topics_rows :
			doc_topics[str(pub_id)].append((topic_id, value))

		# Build topic nodes and paper-topic edges
		topic_nodes = set() 
		topic_paper_edges = set()

		# Retrieve top topics for each document from the db
		topic_ids_per_doc = []
		for doc_id in doc_ids :

#			topics = db.select(fields=["topic_id", "value"], 
#												 table="doc_topics", 
#												 where="paper_id='%s' AND ntopics=%d" % (doc_id, ntopics))

			topics = doc_topics[doc_id]
			if len(topics)>0 :
				topic_ids, topic_values = zip(*topics)

				topic_ids_per_doc.append(topic_ids)
# 				topic_values_per_doc.append(topic_values)

				topic_nodes.update(topic_ids)
				topic_paper_edges.update([(doc_id, topic_ids[t], topic_values[t]) for t in xrange(len(topic_ids))])

# 		for d in xrange(len(doc_ids)) :
# 			topic_ids = topic_ids_per_doc[d]
# 			topic_values = topic_values_per_doc[d]


		# Normalize edge weights with the maximum value
		topic_paper_edges = normalize_edges(topic_paper_edges)

		# From the list of relevant topics f
#		rules = self.get_frequent_topic_pairs(topic_ids_per_doc, min_conf_topics)
		topic_topic_edges = get_rules_by_lift(topic_ids_per_doc, config.MIN_TOPIC_LIFT)

		topic_words_rows = db.select(["topic_id", "words"], "topic_words")
		topic_words = {topic_id:words for topic_id, words in topic_words_rows}

		# Cast topic_nodes to list so we can assure element order
		topics = [(topic_id, topic_words[topic_id].encode("UTF-8")) for topic_id in topic_nodes]

		# Select only the names of the topics being considered here  
		# and store in a class attribute 
# 		topic_names = topic_modeling.get_topic_names(topics, tokens)
# 		self.topic_names = {tid: topic_names[tid] for tid in topic_nodes}

		return topics, topic_topic_edges, topic_paper_edges


	def get_ngrams_layer_db(self, doc_ids):
		'''
		Create words layers by retrieving TF-IDF values from the DB (previously calculated).
		'''
		word_nodes = set()
		paper_word_edges = set()

		ngrams_per_doc = []
		for doc_id in doc_ids :
			rows = db.select(fields=["ngram", "value"], 
											 table="doc_ngrams",
											 where="(paper_id='%s') AND (value>=%f)" % (doc_id, config.MIN_NGRAM_TFIDF))


			if (len(rows) > 0) :
				topic_names, top_values = zip(*rows)
				topic_names = [tw.encode("UTF-8") for tw in topic_names]

				word_nodes.update(topic_names)
				paper_word_edges.update([(doc_id, topic_names[t], top_values[t]) for t in range(len(topic_names))])

				ngrams_per_doc.append(topic_names)

		## TEMPORARY ##
		# PRINT MEAN NGRAMS PER DOC
#		mean_ngrams = np.mean([len(ngrams) for ngrams in ngrams_per_doc])
#		print "%f\t" % mean_ngrams,

		# Get get_rules_by_lift between co-occurring ngrams to create edges between ngrams
		word_word_edges = get_rules_by_lift(ngrams_per_doc, min_lift=config.MIN_NGRAM_LIFT)

		# Normalize edges weights by their biggest value
		word_word_edges = normalize_edges(word_word_edges)
		paper_word_edges = normalize_edges(paper_word_edges)

		return [(word,) for word in word_nodes], word_word_edges, paper_word_edges



	def parse_tfidf_line(self, line) :
		parts = line.strip().split()
		tokens = parts[0::2]
		tfidf  = map(float, parts[1::2])
		return dict(zip(tokens, tfidf))

	
	def include_pubs_atts(self, pubs):

		# Build map for publication years
		rows = db.select(["id", "title", "year"], table="papers")
		years = {}
		titles = {}
		for id, title, year in rows:
			years[id] = year
			titles[id] = title

#		titles = [titles[id].encode("UTF-8") for id in pubs]
#		years =  [years[id] for id in pubs]

		return [(id, titles[id].encode("UTF-8"), years[id]) for id in pubs]


#	@profile
	def dump_graph_to_csvs(self) :
		'''
		Build the full graph by building each layer and then assembling 
		them all together into a complete graph.
		'''

		pub_ids, citation_edges = self.get_papers_layer()
		pubs = self.include_pubs_atts(pub_ids)

#		make_csv("csvs/pub.csv", ["entity_id:ID(pub)", "title:string", "year:int"], pubs)
#		make_csv("csvs/pub-pub.csv", [":START_ID(pub)", ":END_ID(pub)", "weight:float"], citation_edges)
#		log.warn("%d papers and %d citation edges." % (len(pubs), len(citation_edges)))
#
#		# Delete some variables to free some memory
#		del pubs, citation_edges
#	
#		# AUTHOR LAYER
#		authors, coauth_edges, auth_edges =  self.get_authors_layer(pub_ids)
#
#		make_csv("csvs/auth.csv", ["entity_id:ID(author)", "name:string"], authors)
#		make_csv("csvs/auth-auth.csv", [":START_ID(author)", ":END_ID(author)", "weight:float"], coauth_edges)
#		make_csv("csvs/pub-auth.csv", [":START_ID(pub)", ":END_ID(author)", "weight:float"], auth_edges)
#		log.warn("%d authors, %d co-authorship edges and %d authorship edges." % (len(authors), len(coauth_edges), len(auth_edges)))
#
#		# Delete some variables to free memory
#		del authors, coauth_edges, auth_edges
#
#		# TOPIC LAYER
#		topics, topic_topic_edges, pub_topic_edges = self.get_topics_layer_db(pub_ids, config.NTOPICS)
#
#		make_csv("csvs/top.csv", ["entity_id:ID(topic)", "top_words:string"], topics)
#		make_csv("csvs/top-top.csv", [":START_ID(topic)", ":END_ID(topic)", "weight:float"], topic_topic_edges)
#		make_csv("csvs/pub-top.csv", [":START_ID(pub)", ":END_ID(topic)", "weight:float"], pub_topic_edges)
#		
#		log.warn("%d topics, %d topic-topic edges and %d paper-topic edges."
#										% (len(topics), len(topic_topic_edges), len(pub_topic_edges)))
#
#		# Delete some variables to free memory
#		del topics, topic_topic_edges, pub_topic_edges

		# WORD LAYER
		word_nodes, word_word_edges, pub_word_edges = self.get_ngrams_layer_db(pub_ids)

		make_csv("csvs/ngram.csv", ["entity_id:ID(ngram)"], word_nodes)
		make_csv("csvs/ngram-ngram.csv", [":START_ID(ngram)", ":END_ID(ngram)", "weight:float"], word_word_edges)
		make_csv("csvs/pub-ngram.csv", [":START_ID(pub)", ":END_ID(ngram)", "weight:float"], pub_word_edges)

		log.warn("%d word nodes, %d word-word edges and %d pub-word edges." 
									% (len(word_nodes), len(word_word_edges), len(pub_word_edges)))


def main() :

	log.basicConfig(format='%(asctime)s [%(levelname)s] : %(message)s', level=log.WARN)
	
	start = time.time()
	mb = ModelBuilder()
	mb.dump_graph_to_csvs()
	print time.time()-start, "seconds."


if __name__ == '__main__':
	
	main()
	
