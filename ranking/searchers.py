"""
Created on May 11, 2015

@author: luamct
"""

import config
import os
import utils
import model
import networkx as nx
from ranking import ranker
from mymysql.mymysql import MyMySQL
from collections import defaultdict
from pylucene import Index
import cPickle
import sys
import numpy as np

builder = None


def build_graph(query, K, H, min_topic_lift, min_ngram_lift, exclude=[], force=False, save=True, load=False):
	"""
	Utility method to build and return the graph model. First we check if a graph file
	exists. If not, we check if the builder class is already instantiated. If not, we do
	it and proceed to build the graph.
	"""
	global builder
	model_folder = config.IN_MODELS_FOLDER % (config.DATASET, K, H)

	# Creates model folder if non existing
	if not os.path.exists(model_folder):
		os.makedirs(model_folder)

	graph_file = utils.get_graph_file_name(query, model_folder)
	if force or (not os.path.exists(graph_file)):

		if not builder:
			builder = model.ModelBuilder()

		# Builds the graph file
		graph = builder.build(query, K, H, min_topic_lift, min_ngram_lift, exclude)

		# Stores gexf copy for caching purposes
		if save:
			nx.write_gexf(graph, graph_file)

		return graph

	else:
		# A gexf copy already exists in disk. Just load it and return
		# print graph_file
		try:
			graph = nx.read_gexf(graph_file, node_type=int)

		except:
			print "Problem opening '%s'." % graph_file
			sys.exit(1)

	return graph


def get_top_nodes(graph, scores, limit=20, return_type="paper"):
	"""
	Helper method that takes the graph and the calculated scores and outputs a sorted rank
	of the request type of node.
	"""

	# Sort from bigger value to smaller
	scores.sort(key=lambda (k, v): v, reverse=True)

	# Map to integer id for layer (to speed up comparison)
	# rtype_id = layers[return_type]

	ranking = []
	for node, score in scores:
		if (graph.node[node]["type"] == return_type):
			ranking.append((node, graph.node[node]["entity_id"], score))

			if (len(ranking) == limit):
				break

	return ranking


class BaseSearcher:
	def __init__(self, **params):
		self.params = params

	def set_params(self, **params):
		for k, v in params.items():
			self.params[k] = v

	def set_param(self, name, value):
		self.params[name] = value


class Searcher:
	"""
	Basic searcher class for the Multi-Layered method.
	"""

	def __init__(self, **params):
		self.params = params
		self.save = True

	def name(self):
		return "MultiLayered"

	def set_save(self, save):
		self.save = save

	def set_params(self, **params):
		for k, v in params.items():
			self.params[k] = v

	def set_param(self, name, value):
		self.params[name] = value

	def number_of_nodes(self):
		""" Return number of nodes if search was called once. """
		return self.nnodes

	def get_nx_graph(self):
		""" Return networkx graph structured if search method was called once. """
		return self.graph


	def search(self, query, exclude=[], limit=20, rtype="paper", force=False):
		"""
		Checks if the graph model already exists, otherwise creates one and
		runs the ranking on the nodes.
		"""
		graph = build_graph(query,
							self.params['K'],
							self.params['H'],
							self.params['min_topic_lift'],
							self.params['min_ngram_lift'],
							exclude, force, load=True, save=self.save)

		# Store number of nodes for checking later
		self.nnodes = graph.number_of_nodes()

		# Rank nodes using subgraph
		scores = ranker.rank_nodes(graph, limit=limit, return_type=rtype, **self.params)

		# Adds the score to the nodes and writes to disk. A stupid cast
		# is required because write_gexf can't handle np.float64
		scores = {nid: float(score) for nid, score in scores.items()}
		nx.set_node_attributes(graph, "score", scores)

		# nx.write_gexf(graph, utils.get_graph_file_name(model_folder, query))

		# Returns the top values of the type of node of interest
		results = get_top_nodes(graph, scores.items(), limit=limit, return_type=rtype)

		# Add to class object for future access
		self.graph = graph

		return [str(pub_id) for _nid, pub_id, _score in results]


class GoogleScholarSearcher:
	"""
	Results from Google Scholar that were previously collected, processed
	and stored in disk. The folder is provided in the constructor.
	"""

	def __init__(self):
		folder = "%s/scholar/%s" % (config.DATA, config.DATASET)

		self.results = {}
		for file_name in os.listdir(folder):

			file_path = os.path.join(folder, file_name)
			with open(file_path, 'r') as file:

				# First line is the query, the others are the results from GoogleScholar
				query = file.readline().strip()

				# Use this set to remove duplicate, although slightly
				# different text for the same publication is very possible
				unique = set()
				titles = []

				for line in file:

					# Only add if non existing yet. It's common for Scholar to return
					# duplicate results
					lowered = line.strip().lower()
					if lowered not in unique:
						unique.add(lowered)
						titles.append(line.strip())

				# Now add to a class object
				self.results[query] = titles

	@staticmethod
	def name():
		return "GoogleScholar"

	def search(self, query, exclude=[], get_titles=False, limit=20, force=False):
		if query not in self.results:
			print "%s only works for 'Manual' query set. Ignoring query '%s'." % (self.name(), query)
			return []

		return list(self.results[query])


class ArnetMinerSearcher:
	"""
	Results from ArnetMiner that were previously collected, processed
	and stored in disk. Right now we are only storing for the manual
	set, because it doesn't make sense to evaluate for the other sets.
	"""

	def __init__(self):
		folder = "%s/aminer/%s/" % (config.DATA, config.DATASET)

		self.results = {}
		for file_name in os.listdir(folder):

			file_path = os.path.join(folder, file_name)
			with open(file_path, 'r') as file:

				# First line is the query, the others are the results from GoogleScholar
				query = file.readline().strip()

				# Use set to remove duplicate, although slightly different
				# text for the same publication is very possible
				self.results[query] = set()
				for line in file:
					self.results[query].add(line.strip().lower())


	def name(self):
		return "ArnetMiner"

	def search(self, query, exclude=[], limit=20, force=False):
		if (query not in self.results):
			print "%s only works for 'Manual' query set. Ignoring query '%s'." % (self.name(), query)
			return []

		return list(self.results[query])


class CiteseerSearcher:
	"""
	Results from CiteseerX that were previously collected, processed
	and stored in disk. The folder is provided in the constructor.
	"""

	def __init__(self, folder):
		self.folder = folder

	def name(self):
		return "Citeseer"

	def search(self, query, limit=20):
		docs = []
		file_name = query.replace(" ", "+") + ".txt"
		with open(os.path.join(self.folder, file_name), 'r') as file:
			for line in file:
				doc_id, _title = line.split('\t')
				docs.append(doc_id)

		return docs[:limit]


class PageRankFilterBeforeSearcher():
	"""
	Ranks using a simple PageRank algorithm in the unweighted citation
	network. To account for the query, only the documents that contain
	at least one term of the query are included in the citation network
	prior to running the PageRank.
	"""

	def __init__(self):
		self.index = Index(config.INDEX_PATH)

		# Get all possible edges
		self.edges = model.get_all_edges()


	def name(self):
		return "PageRank(pre)"

	def search(self, query, exclude=[], force=False, limit=20):

		# Fetches all document that have at least one of the terms
		pubs = self.index.search(query,
								 search_fields=["title", "abstract"],
								 return_fields=["id"],
								 ignore=exclude)

		# Unpack and convert to a set for fast lookup
		pubs = set([pub_id for (pub_id,) in pubs])

		# index_ids, _scores = self.index.search(query, ["title", "abstract"], limit=limit, mode="ALL")
		# docs = set(self.index.get_documents(index_ids, "id"))

		g = nx.DiGraph()
		for u, v in self.edges:
			if (u in pubs) and (v in pubs):
				g.add_edge(u, v)

			#		print "PageRank with %d nodes." % g.number_of_nodes()
		r = nx.pagerank(g, alpha=0.7)

		if len(r) == 0:
			return []

		ids, _pg = zip(*sorted(r.items(), key=lambda (k, v): v, reverse=True))
		return ids[:limit]


class PageRankFilterAfterSearcher():
	"""
	Ranks using a simple PageRank algorithm in the unweighted citation
	network. To account for the query, after running the page rank, the
	top values WHICH CONTAIN at least one term of the query are used as
	the result list.
	"""

	def __init__(self):
		self.index = Index(config.INDEX_PATH)

		# Checks if the full graph for this dataset was already ranked.
		# If not, run page rank and store the results
		pr_file_path = "%s/page_rank/%s.p" % (config.DATA, config.DATASET)
		if not os.path.exists(pr_file_path):
			g = nx.DiGraph()
			g.add_edges_from(model.get_all_edges())

			print "Running pageRank with %d nodes." % g.number_of_nodes()
			self.pr = nx.pagerank(g)

			cPickle.dump(self.pr, open(pr_file_path, "w"))

		# Else, just loads it
		else:
			self.pr = cPickle.load(open(pr_file_path, 'r'))


	def name(self):
		return "PageRank(pos)"

	def search(self, query, force=False, exclude=[], limit=20):

		# Sorts documents decreasingly by page rank value
		ids, _values = zip(*sorted(self.pr.items(), key=lambda (k, v): v, reverse=True))

		# Fetches all document that have at least one of the terms
		pubs = self.index.search(query,
								 search_fields=["title", "abstract"],
								 return_fields=["id"],
								 ignore=exclude)

		# Unpack and convert to a set for fast lookup
		pubs = set([pub_id for (pub_id,) in pubs])

		results = []
		for id in ids:
			if id in pubs:
				results.append(id)
				if len(results) == limit:
					break

		return results


class PageRankSubgraphSearcher:
	"""
	Ranks by the most cited documents included in the induced subgraph
	(see ModelBuilder for details).
	"""

	def __init__(self, **params):
		self.params = params

	def name(self):
		return "PageRank(G)"

	def search(self, query, exclude=[], limit=50, force=False):

		graph = build_graph(query,
							self.params['K'],
							self.params['H'],
							self.params['min_topic_lift'],
							self.params['min_ngram_lift'],
							exclude, force, load=True)


		# Simple method to check if node is a document node.
		is_doc = lambda node: node["type"] == "paper"

		# Builds a new unweighted graph with only the documents as nodes
		docs_graph = nx.DiGraph()

		# Removes all non doc nodes
		for u, v in graph.edges():
			u = graph.node[u]
			v = graph.node[v]
			if is_doc(u) and is_doc(v):
				docs_graph.add_edge(u["entity_id"], v["entity_id"])

		r = nx.pagerank(docs_graph, alpha=0.7)
		if len(r) == 0:
			return []

		ids, _pg = zip(*sorted(r.items(), key=lambda (k, v): v, reverse=True))
		return ids[:limit]


class TopCitedSubgraphSearcher(BaseSearcher):
	"""
	Ranks by the most cited documents included in the induced subgraph
	(see ModelBuilder for details).
	"""

	def __init__(self, **params):
		self.params = params

	def name(self):
		return "TopCited(G)"

	def search(self, query, exclude=[], limit=50, force=False):

		graph = build_graph(query,
							self.params['K'],
							self.params['H'],
							self.params['min_topic_lift'],
							self.params['min_ngram_lift'],
							exclude, force, load=True)

		ncits = defaultdict(int)
		is_pub = lambda n: (graph.node[n]["type"] == "paper")
		for n1, n2 in graph.edges():
			if is_pub(n1) and is_pub(n2):
				ncits[graph.node[n2]["entity_id"]] += 1

		# Sort dict by value
		sorted_ncits = sorted(ncits.items(), key=(lambda (k, v): v), reverse=True)
		ids, _cits = zip(*sorted_ncits)
		return ids[:limit]


class TopCitedGlobalSearcher:
	"""
	Ranks by the most cited included all documents that contain all
	the keywords in the query.
	"""

	def __init__(self):
		self.index = Index(config.INDEX_PATH)

		# Get citation counts and store into dict for fast lookup
		db = MyMySQL(db=config.DB_NAME, user=config.DB_USER, passwd=config.DB_PASSWD)

		ncitations = db.select_query("SELECT cited, COUNT(*) from graph GROUP BY cited")
		self.ncitations = dict(ncitations)

	def name(self):
		return "TopCited"

	def search(self, query, exclude=[], limit=50, force=False):

		# Fetches all document that have at least one of the terms
		docs = self.index.search(query,
								 search_fields=["title", "abstract"],
								 return_fields=["id"],
								 ignore=exclude)

		# docs = self.index.get_documents(index_ids, "id")

		# print "%d documents found." % len(docs)
		ncitations = []
		for (doc_id,) in docs:

			if doc_id in (self.ncitations):
				ncitations.append((self.ncitations[doc_id], doc_id))

		# Sort by number of citations and returned top entries
		_citations, ids = zip(*sorted(ncitations, reverse=True))
		return ids[:limit]


class TFIDFSearcher():
	"""
	Returns the top tf-idf scored documents according to the query.
	"""

	def __init__(self):
		self.index = Index(config.INDEX_PATH)

	def name(self):
		return "TF-IDF"

	def search(self, query, exclude=[], limit=50, force=False):
		# Fetches all document that have at least one of the terms
		pub_ids = self.index.search(query,
									search_fields=["title", "abstract"],
									return_fields=["id"],
									ignore=exclude)

		# Filter top n_starting_nodes
		return [pub_id for (pub_id,) in pub_ids]


class BM25Searcher():
	"""
	Returns the top tf-idf scored documents according to the query.
	"""

	def __init__(self):
		self.index = Index(config.INDEX_PATH, similarity="BM25")

	def name(self):
		return "BM25"

	def search(self, query, exclude=[], limit=50, force=False):
		# Fetches all document that have at least one of the terms
		pub_ids = self.index.search(query,
									search_fields=["title", "abstract"],
									return_fields=["id"],
									ignore=exclude)

		# Filter top n_starting_nodes
		return [pub_id for (pub_id,) in pub_ids]


class CiteRankSearcher():
	"""
	Ranks using the CiteRank variant, which is basically a PageRank, but it
	includes a teleportation array defined by the age of each paper. Older
	papers are less likely to be randomly visited by a walker. To account
	for the query, after running the page rank, the top values WHICH CONTAIN
	at least one term of the query are used as the result list.
	"""

	def __init__(self, tau, filter_before=True):
		self.index = Index(config.INDEX_PATH)
		self.tau = tau

	def name(self):
		return "CiteRank"

	def search(self, query, exclude=[], limit=20, force=False):

		# import warnings
		# warnings.filterwarnings('error')

		file_path = config.CITERANK_FILE_PATH
		if not os.path.exists(file_path):
			g = nx.DiGraph()
			g.add_edges_from(model.get_all_edges())

			# Remove documents from the exclude list
			g.remove_nodes_from(exclude)

			# Get year of each paper for assembling personalization array next
			db = MyMySQL(db=config.DATASET)
			rows = db.select(["id", "year"], table="papers")
			years = {}
			for pub_id, year in rows:
				if year is not None:
					years[pub_id] = year

			# Calculate the median to use in the missing values
			year_median = np.median(years.values())

			# Create a personalization array by exponentially decaying
			# each paper's factor by its age
			pers = {}
			for node in g.nodes():
				if (node not in years) or (years[node] < 1960) or (years[node] > 2013):
					years[node] = year_median

				pers[node] = np.exp(float(years[node] - 2013) / self.tau)
			#				try :
			#				except Warning:
			#					print "Warning!"
			#					print node
			#					print year
			#					print

			print "Running PageRank with %d nodes and age defined personalization vector." % g.number_of_nodes()
			r = nx.pagerank(g, personalization=pers)

			print "Writing results"
			cPickle.dump(r, open(file_path, "w"))


		# Loads cached page rank values for every node
		r = cPickle.load(open(file_path, "r"))

		# Sorts documents decreasingly by page rank value
		ids, _score_ = zip(*sorted(r.items(), key=lambda (k, v): v, reverse=True))

		# Fetches all document that have at least one of the terms.
		# Store them in a set for fast lookup
		pub_ids = self.index.search(query,
									search_fields=["title", "abstract"],
									return_fields=["id"],
									ignore=exclude)

		pub_ids = set([pid for (pid,) in pub_ids])

		results = []
		for id in ids:
			if id in pub_ids:
				results.append(id)
				if len(results) == limit:
					break

		return results


class WeightedTopCitedSubgraphSearcher(BaseSearcher):
	"""
	Ranks by the weighted citations included in the induced subgraph
	(see ModelBuilder for details).
	"""

	def __init__(self, **params):
		self.params = params

	def name(self):
		return "WeightedTopCited(G)"

	def set_params(self, **params):
		for k, v in params.items():
			self.params[k] = v

	def set_param(self, name, value):
		self.params[name] = value

	def search(self, query, exclude=[], limit=50, force=False, yc=2013):

		graph = build_graph(query,
							self.params['K'],
							self.params['H'],
							self.params['min_topic_lift'],
							self.params['min_ngram_lift'],
							exclude, force, load=True)


		# Remove isolated nodes
		graph.remove_nodes_from(nx.isolates(graph))

		# Simple method to check if node is a document node.
		is_pub = lambda n: (graph.node[n]["type"] == "paper")

		# Get year median to replace missing values.
		# npapers = 0
		years = []
		for u in graph.nodes() :
			if is_pub(u) :
				# npapers += 1
				if (graph.node[u]["year"] > 0) :
					years.append(graph.node[u]["year"])

		year_median = np.median(years)

		yo = 1950
		wcits = defaultdict(float)
		for u in graph.nodes():
			if is_pub(u):
				weighted_citation = 0
				nc = 0 # citation count

				year = graph.node[u]["year"]
				if year == 0: # missing
					year = year_median
				elif year < yo or year > yc: # truncate
					year = max(min(year, yc), yo)
				age_decay = np.exp(-self.params["age_relev"]*(yc-year))

				query_sim = np.exp(-self.params["query_relev"]*(1.0-graph.node[u]["query_score"]))

				in_edges = graph.in_edges(u, data=True)
				for v, _, atts in in_edges:
					if is_pub(v):
						ctx_sim = np.exp(-self.params["ctx_relev"]*(1.0-atts["weight"]))
						weighted_citation += ctx_sim*query_sim*age_decay
						nc += 1

				if nc > 0:
					weighted_citation = weighted_citation*nc**(-self.params["beta"])
				wcits[graph.node[u]["entity_id"]] = weighted_citation

		# Sort dict by value
		sorted_wcits = sorted(wcits.items(), key=(lambda (k, v): v), reverse=True)
		ids, _wcits = zip(*sorted_wcits)
		return ids[:limit]


if __name__ == '__main__':
	gs = ArnetMinerSearcher()
	r = gs.search("sentiment analysis", limit=10)
	print "\n".join(sorted(r))


# db = MyMySQL(db='csx')
# s = Searcher(**config.PARAMS)
#	pub_ids = s.search("subgraph mining", limit=20)
#	for id in pub_ids :
#		print "%12s\t %s" % (id, db.select_one("title", table="papers", where="id='%s'"%id))

#	for id in pub_ids :
#		print id




