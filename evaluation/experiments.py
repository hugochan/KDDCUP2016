#!/home/luamct/anaconda/bin/python -W ignore::DeprecationWarning
'''
Created on Sep 16, 2014

@author: luamct
'''
from mymysql.mymysql import MyMySQL
import config
from evaluation.query_sets import load_query_set
from baselines.meng import MengSearcher
from ranking.searchers import Searcher, PageRankSubgraphSearcher,\
	TopCitedSubgraphSearcher, TopCitedGlobalSearcher, TFIDFSearcher, BM25Searcher,\
	PageRankFilterBeforeSearcher, PageRankFilterAfterSearcher,\
	GoogleScholarSearcher, ArnetMinerSearcher, CiteRankSearcher, WeightedTopCitedSubgraphSearcher
from collections import defaultdict
import time
import numpy as np
from config import PARAMS

import logging as log
import os
import cPickle
from baselines.scholar import match_by_title
from evaluation.metrics import apk, ndcg, recall_at, precision_at
#warnings.filterwarnings('error')


log.basicConfig(format='%(asctime)s [%(levelname)s] : %(message)s', level=log.INFO)

db = MyMySQL(db=config.DB_NAME, user=config.DB_USER, passwd=config.DB_PASSWD)


def get_year(paper_id):
	year = db.select_one("year", table="papers", where="id='%s'"%paper_id)
	return year


def filter_newer_entries(year, results, n) :
	valid = []
	for doc in results :
		cit_year = get_year(doc)
		if cit_year and (cit_year < year) :
			valid.append(doc)

			if len(valid) > n:
				break

	return valid


def vary_rho_values(searcher, queries, rho_param, rho_values) :
	'''
	Varies one given rho value and sets the others uniformly so
	that all sum to one. Its meant to find optimal values for each
	layer.
	'''

	# Rho parameters except the varying one
	rho_params = ['papers_relev', 'authors_relev', 'venues_relev', 'words_relev']

	print "\nVarying parameter '%s'" % rho_param
	for rho_value in rho_values :

		# Set rho values equally such that all sum (plus rho_value) sum up to 1
		other_rhos = (1.0-rho_value)/(len(rho_params)-1)
		for rp in rho_params :
			searcher.set_param(rp, other_rhos)

		# Now overwrite the varying parameter accordingly
		searcher.set_param(rho_param, rho_value)

		print "%.2f\t%.3f\t" % (rho_value, other_rhos),
		get_search_metrics(queries, searcher, force=False)



def show_layers_effect(query_set, searcher, layers_combs):
	'''
	Shows performance for different layers configuration.
	'''

	queries = load_query_set(query_set, 40)

	legend = {'P': 'papers_relev',
						'A': 'authors_relev',
						'K': 'words_relev',
						'V': 'venues_relev'}

	params = {'age_relev':   0.0,
						'query_relev': 0.0,
						'ctx_relev':   0.0}

	for layers in layers_combs :

		print "%s\t" % layers,

		for layer in legend:
			params[legend[layer]] = 1.0 if (layer in layers) else 0.0

		searcher.set_params(**params)
		get_search_metrics(queries, searcher, show=True, force=True,
											 results_file=("%s/results/layers_effect/%s.p" % (config.DATA, layers)))

	print


def show_attenuators_effect(queries, searcher):
	'''
	Shows performance for different layers configuration.
	'''

	legend = {'A': 'age_relev',
						'C': 'ctx_relev',
						'Q': 'query_relev'}

	zeros = {'age_relev':   0.0,
						'query_relev': 0.0,
						'ctx_relev':   0.0}

	atts_combs = ['A', 'C', 'Q', 'AC', 'CQ', 'AQ', 'ACQ']
	for atts in atts_combs :

		print "%s\t" % atts,

		# First zero parameters of interest, then sets values for the
		# ones present in the current configuration
		searcher.set_params(**zeros)
		for att in atts:
			searcher.set_param(legend[att], PARAMS[legend[att]])

		get_search_metrics(queries, searcher, show=True,
											 results_file=(config.DATA + "results/atts/" + atts))

	print


def vary_parameters(searcher, queries, name, values) :
	'''
	Vary a single parameter to find optimal values.
	'''

	# Force subgraph rebuilt if varying parameter affects the subgraph
	force = (name in set(['K', 'H', 'min_ngram_conf', 'min_topic_conf']))
#	force = True

	print "\nVarying parameter '%s'" % name
	for value in values :
		searcher.set_param(name, value)
		print "%s\t" % str(value),
		get_search_metrics(queries, searcher, force=force)



def show_results(query_id, results, right) :
	right = set(right)
	print query_id
	for pub_id in results :
		print ("OK\t" if pub_id in right else "  \t"),
		print "%16s\t%s" %(pub_id, db.select_one("title", table="papers", where="id='%s'"% pub_id))


def save_results(results, file_path) :
	'''
	Saves the results in a output file.
	'''
	cPickle.dump(results, open(file_path, 'w'))


def filter_missing(ids, relevs, titles) :
	fids = []
	frelevs = []
	ftitles = []
	for i in xrange(len(ids)) :
		if (ids[i] != '') :
			fids.append(ids[i])
			frelevs.append(relevs[i])
			ftitles.append(titles[i])

	return fids, frelevs, ftitles


def get_search_metrics(truth, searcher, show=True, force=False, results_file=None) :
	'''
	Run searches on each survey (query -> ground truth) and return
	the evaluate metric for each instance. Right now the metrics being
	returned are MAP, P@5, P@10, P@20.

	Returns: dict {metric: array of values}
	'''
	metrics = defaultdict(list)
	results = []

	for query, doc_id, year, correct_ids, relevs, correct_titles in truth :

#		print "Processing '%s'" % query
		start = time.time()

		returned_ids = searcher.search(query, exclude=[doc_id], force=force, limit=100)

		# GoogleScholar is treated differently, since neither the scholar
		# returned documents, nor the pubs from the manual set are in our databases.
		# We match the documents by text similarity and create string ids so that
		# the rest of the evaluation can remain the same. Note that the correct_ids and
		# results values are replaced in the process
		if (searcher.name()=="GoogleScholar") or (searcher.name()=="ArnetMiner") :
			correct_ids, returned_ids = match_by_title(query, returned_ids, correct_titles)

		else:

			# If not Scholar or AMiner, also remove pubs that are more recent than the
			# query pub, and therefore couldn't be used as a viable candidate
			if (year != "") :
				returned_ids = filter_newer_entries(int(year), returned_ids, 20)
			else :
				returned_ids = returned_ids[:20]

		# Filter pubs not found in the dataset, which can't be considered false negatives
		correct_ids, relevs, correct_titles = filter_missing(correct_ids, relevs, correct_titles)

		# Store each query results for saving into disk
		results.append( (correct_ids, relevs, returned_ids) )

#		show_results(doc_id, results, correct_ids)
#		metrics["NN"].append(searcher.number_of_nodes())
		metrics["Time"].append((time.time()-start))

#		global builder
#		metrics["NWords"].append(searchers.builder.nwords)
#		metrics["topics_dens"].append(builder.topic_density)
#		metrics["ngrams_dens"].append(builder.ngram_density)

#		print apk(correct_ids, returned_ids, k=20)

		metrics["MAP"].append( apk(correct_ids, returned_ids, k=20) )
		metrics["P@5"].append( precision_at(correct_ids, returned_ids, k=5) )
		metrics["P@10"].append( precision_at(correct_ids, returned_ids, k=10) )
		metrics["P@20"].append( precision_at(correct_ids, returned_ids, k=20) )
		metrics["R@20"].append( recall_at(correct_ids, returned_ids, k=20) )
		metrics["NDCG@20"].append( ndcg(correct_ids, returned_ids, relevs, 20) )


	if results_file :
		save_results(results, results_file)

	if show:
		for m in ["MAP", "P@5", "P@10", "P@20", "R@20", "NDCG@20", "Time"] :
			print u"%.1f \xb1 %.1f\t" % (100*np.mean(metrics[m]), np.std(metrics[m])),
		print

	return metrics


def get_results_file(query_set, method_name) :
	folder = "%s/results/%s/%s" % (config.DATA, config.DATASET, query_set)
	if not os.path.exists(folder) :
		os.makedirs(folder)

	return "%s/%s.p" % (folder, method_name)


def get_layer_results(queries, searcher, folder, layer) :

	db = MyMySQL(db=config.DATASET)

	def get_pub(pub_id) :
		return db.select_one("title", table="papers", where="id='%s'" % pub_id)

	def get_author(author_id) :
		return db.select_one("name", table="authors", where="cluster=%s" % author_id)

	def get_venue(venue_id):
		abbrev, name = db.select_one(["abbrev", "name"], table="venues", where="id=%s" % venue_id)
		return " ".join((abbrev, name)).strip()

	def get_keyword(kw):
		return kw

	# Create the folder that will hold the results for this layer
	if not os.path.exists(folder) :
		os.makedirs(folder)

	print "\n%s" % folder

	# Each layer has a different handler to get the name of the entity
	get_entities = {'paper': get_pub,
									'author': get_author,
									'venue': get_venue,
									'ngram': get_keyword}

	# Now fetch the results and save them
	for query in queries :

		file_path = os.path.join(folder, query.replace(' ', '+') + ".txt")
		print " ", query

		entity_ids = searcher.search(query, rtype=layer, limit=50)
		with open(file_path, 'w') as file :
			for eid in entity_ids:
				name = get_entities[layer](eid).strip()
				print >> file, "%s" % (name.encode("UTF-8"))


def save_layers_results_query_set(query_set) :

	queries = load_query_set(query_set, 10)
	queries = [query for query,_,_,_,_,_ in queries]

	layers = ['paper',
						'author',
						'venue',
						'ngram']

	searcher = Searcher(**PARAMS)

	for layer in layers :

		folder = "%s/results/layers/%s/%s/" % (config.DATA, config.DATASET, layer)
		get_layer_results(queries, searcher, folder, layer)


def save_layers_results_queries(queries, folder) :

	layers = ['paper',
						'author',
						'venue',
						'ngram']

	searcher = Searcher(**PARAMS)

	for layer in layers :
		get_layer_results(queries, searcher, os.path.join(folder, layer), layer)


def check_topics_effect(searcher, query_set) :

	queries = load_query_set(query_set, 10)
	get_search_metrics(queries, searcher, force=True)


def time_diversity(names, query_set) :


	# Get year of each paper for assembling personalization array next
	db = MyMySQL(db=config.DATASET)
	rows = db.select(["id", "year"], table="papers", where="year is not NULL and year between 1950 and 2013")
	years = {pub_id: year for pub_id, year in rows}

	for name in names :

		file_path = "%s/results/%s/%s/%s.p" % (config.DATA, config.DATASET, query_set, name)

		returned_years = []
		results = cPickle.load(open(file_path, 'r'))
		for _correct, _relevances, returned in results :
			for r in returned :
				if r in years :
					returned_years.append(years[r])

		print "%s\t%.2f\t%.2f" % (name, np.mean(returned_years), np.std(returned_years))


#			print correct, relevances, returned
#			break



def main() :

#	for qs in ["manual", "surveys", "testing"] :
#		print "\n%s" % qs
#		time_diversity(["MultiLayered", "TopCited(G)"], qs)

#	sys.exit()

#	config.IN_MODELS_FOLDER = config.DATA + "topic_models/%s_%d_%d"
##	check_topics_effect(Searcher(**PARAMS), "manual")
#	show_layers_effect("manual", Searcher(**PARAMS), ["PAK", "PAT", "PAKT"][:1])

#	save_layers_results_queries(["citation recommendation",
#															 "author recommendation",
#															 "link prediction"],
#															 folder="/var/tmp/results")
#	save_layer_results("manual")

#	query_set = 'manual'
#	query_set = 'surveys'
#	query_set = 'tuning'
#	query_set = 'testing'
#	queries = load_query_set(query_set, 200)

#	vary_parameters(Searcher(**PARAMS), queries, 'age_relev', [0.0, 0.001, 0.01, 0.1, 0.25, 0.5, 1.0])
#	vary_parameters(Searcher(**PARAMS), queries, 'ctx_relev', [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
#	vary_parameters(Searcher(**PARAMS), queries, 'query_relev', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8])

#	layers = ["P","PA","PV","PK","PAV","PAK","PKV","PAKV"]
#	show_layers_effect(queries, Searcher(**BEST_PARAMS), layers)

#	show_attenuators_effect(queries, Searcher(**PARAMS))

#	vary_rho_values(Searcher(**PARAMS), queries, 'papers_relev',  [0.05, 0.1, 0.25, 0.5, 0.75, 0.85, 0.9, 1.0])
#	vary_rho_values(Searcher(**PARAMS), queries, 'authors_relev', [0.0, 0.1, 0.25, 0.5, 0.75, 0.9])
#	vary_rho_values(Searcher(**PARAMS), queries, 'topics_relev',  [0.0, 0.1, 0.25, 0.5, 0.75, 0.9])
#	vary_rho_values(Searcher(**PARAMS), queries, 'words_relev',   [0.0, 0.1, 0.25, 0.5, 0.75, 0.9])
#	vary_rho_values(Searcher(**BEST_PARAMS), queries, 'venues_relev',  [0.0, 0.05, 0.1, 0.25, 0.5])

#	get_other_layers(load_query_set('manual', 10), Searcher(**BEST_PARAMS), "ngram")


	query_sets = [
							# 'manual',
							# 'surveys',
							# 'tuning',
							'testing'
							]

	searchers = [
						Searcher(**PARAMS),
						# Searcher(**config.PARAMS),
						# PageRankSubgraphSearcher(**PARAMS),
						# TopCitedSubgraphSearcher(**PARAMS),
						# TopCitedGlobalSearcher(),
						# TFIDFSearcher(),
						# BM25Searcher(),
						# CiteRankSearcher(tau=2.6),
						# PageRankFilterBeforeSearcher(),
						# PageRankFilterAfterSearcher(),
#						GoogleScholarSearcher(),
#						ArnetMinerSearcher(),
						#MengSearcher(),
#						CiteseerSearcher("eval/citeseer"),
						# WeightedTopCitedSubgraphSearcher(**PARAMS)
					]

	for query_set in query_sets :

		log.info("Running '%s' query set.\n" % query_set)

		queries = load_query_set(query_set, 200)
		for s in searchers :
			print "%s\t" % s.name(),
#			print "\nRunning %s with %d queries from %s set..." % \
#																	(s.name(), len(queries), query_set)
			if s.name() == "MultiLayered":
				s.set_params(**{
							  'K': 20,
						      'H': 1,
							  'papers_relev': 0.25,
							  'authors_relev': 0.25,
						   	  'words_relev': 0.25,
							  'topics_relev' : 0.0,
							  'venues_relev': 0.25,
							  'alpha': 0.3,
							  'query_relev': 0.3,
							  'age_relev': 0.01,
   							  'ctx_relev': 0.5})

			if s.name() == "TopCited(G)": # TopCitedSubgraphSearcher
				s.set_params(**{
							  'K': 20,
						      'H': 1,
							  'papers_relev': 0.25,
							  'authors_relev': 0.25,
						   	  'words_relev': 0.25,
							  'topics_relev' : 0.0,
							  'venues_relev': 0.25,
							  'alpha': 0.3,
							  'query_relev': 0.3,
							  'age_relev': 0.01,
   							  'ctx_relev': 0.5})


			if s.name() == "WeightedTopCited(G)":
				s.set_params(**{
							  'K': 20,
						      'H': 1,
							  'query_relev': 0.15,  # 0.15
						      'age_relev': 0.01, # 0.01
							  'ctx_relev': 0.8, # 0.6 (manual), 0.8
							  'beta': 0.1}) # 0.1
			rfile = get_results_file(query_set, s.name())
			get_search_metrics(queries, s, force=True, results_file=rfile)
			del s



if __name__=="__main__":
	main()

