'''
Created on Jun 2, 2015

@author: luamct
'''
from mymysql.mymysql import MyMySQL
from utils import PubTexts, get_texts
from collections import defaultdict
from ranking.searchers import BaseSearcher
from sklearn.feature_extraction.text import CountVectorizer
from evaluation.query_sets import load_query_set
from operator import itemgetter
import config
import subprocess as sp
import numpy as np
import sys
import os

db = MyMySQL(db='csx', user='root', passwd='')

DATA_PATH = config.DATA + "baselines/link_plsa_lda/%s"
BASE_PATH = "/home/luamct/ca/others/link_plsa_lda/"


class LinkPLSASearcher(BaseSearcher) :
	
	def __init__(self, query_set, ntopics) :

		data_path = DATA_PATH % query_set
		self.results = read_output(data_path, 60)

	def search(self, pub_id, exclude=[], limit=20, force=False):
		return self.results[pub_id]


def print_row(file, row):
	''' Writing terms frequency. '''
	row = row.tocoo()
	entries = zip(row.col, row.data)
	line = ' '.join(["%d:%d" % (idx, value) for idx, value in entries])
	print >> file, len(entries), line


def make_input_file(folder, test_ids=[]) :

	# Create folder if necessary
	if not os.path.exists(folder):
		os.mkdir(folder)

	# Sample some publications and create the structure to 	
	pubs = PubTexts(n=100000)
	pub_ids = set(pubs.ids())

	# Maps string id to incremental integer cited_ids_map
#	for pid in pubs.cited_ids_map():
#		cited_ids_map[str(pid)] = len(cited_ids_map)

	# Ignore these cited_ids_map in the training files so they can be used for testing
	test_ids_set = set(test_ids)

	# Make citation file
	cited_ids_map = {}
	citations = defaultdict(list)
	rows = db.select(fields=["citing", "cited"], table="graph")
	for citing, cited in rows:

		# Convert to ascii string		
		citing, cited = str(citing), str(cited)

		# Only include it if it's on sample set and it's not on test set
		if (citing in pub_ids) and (cited in pub_ids) and \
			 (citing not in test_ids_set) and (cited not in test_ids_set):

#			print "Adding"
			if cited not in cited_ids_map:
				cited_ids_map[str(cited)] = len(cited_ids_map)

			citations[str(citing)].append(cited_ids_map[str(cited)])

	cits_per_pub = np.mean([len(c) for c in citations.values()])
	print "%d citing pubs with %.2f citations per pub." % (len(citations), cits_per_pub)
	print "%d cited pubs." % len(cited_ids_map)

	citing = citations.keys()
#	citing_mask = [(pid in citing) for pid in pubs.ids()]

	vec = CountVectorizer(stop_words='english', ngram_range=(1,2), 
												max_df=0.5, min_df=5,
												max_features=20000)
	texts = vec.fit_transform(pubs.texts(citing, use_title=True, use_abs=True))

	# Write training file. Every row in the citing_tr.txt is the vector 
	# representation of the text in the file.
	citations_file = open(os.path.join(folder, "citations_tr.txt"), "w")
	citing_file = open(os.path.join(folder, "citing_tr.txt"), "w")
	citing_ids_file = open(os.path.join(folder, "citing_tr_ids.txt"), "w")

	for i, pid in enumerate(citing) :

		# Writing citing ids to map back on the search
		print >> citing_ids_file, str(pid)

		# Writing citations
		cited = sorted(citations[str(pid)])
		print >> citations_file, len(cited), ' '.join(map(str, cited))

		print_row(citing_file, texts[i])

	citations_file.close()
	citing_file.close()
	citing_ids_file.close()

	# Release some memory
	del citations, citing, texts

	# Find vector representation for texts in the test set. This one we can't 
	# fit, only transform, because it's test data (therefore unseen).
	texts = vec.transform(get_texts(test_ids, use_title=True, use_abs=True))

	# Write the test files now. The citing contains term frequencies and the citations
	# is basically zeros, since it's actually what we are trying to predict.
	citations_file = open(os.path.join(folder, "citations_ts.txt"), "w")
	citing_file = open(os.path.join(folder, "citing_ts.txt"), "w")
	citing_ids_file = open(os.path.join(folder, "citing_ts_ids.txt"), "w")

	for i in xrange(len(test_ids)) :

		print >> citing_ids_file, str(test_ids[i])
		print >> citations_file, '0'
		print_row(citing_file, texts[i])

	citations_file.close()
	citing_file.close()


	# Sort numeric cited_ids_map so that each line x corresponds to pub x
	str_ids, num_ids = zip(*cited_ids_map.items())
	cited = np.asarray(str_ids)[list(num_ids)]
	texts = vec.fit_transform(pubs.texts(cited, use_title=True, use_abs=True))

	cited_file = open(os.path.join(folder, "cited.txt"), "w")
	cited_ids_file = open(os.path.join(folder, "cited_ids.txt"), "w")

	# Writing terms frequency
	for i in xrange(len(cited)) :

		# Writing cited ids to map back the searches
		print >> cited_ids_file, str(cited[i])

		print_row(cited_file, texts[i])

	cited_file.close()
	cited_ids_file.close()
	
	print "Done!"


def read_output(data_path, ntopics, max):
	
	# Quick utility method to read ids' files
	def read_ids_file(path):
		with open(path, 'r') as file :
			return [str(line.strip()) for line in file]

	# Read cited ids and citing ids files to to map numeric ids to string ids
	cited_ids  = read_ids_file(os.path.join(data_path, "cited_ids.txt")) 
	citing_ids = read_ids_file(os.path.join(data_path, "citing_ts_ids.txt"))

	ncited = len(cited_ids)
	nciting = len(citing_ids)

	print ncited, nciting, ncited*nciting

	# Results are shown as a dict (pub_id : list of pub_ids)
	results = defaultdict(list)

	# Open and interpret output file
	output_path = os.path.join(data_path, "output", "k%d"%ntopics, "pdt-predictions.dat")  
	with open(output_path, 'r') as file:

		# There's an entry for each possible pair (citing,cited) with a 
		# binary variable determining if there's a citation between them
		for i in xrange(nciting) :

			row = []
			for j in xrange(ncited) :

				line = file.readline()
				citing_id, cite, value = line.strip().split()
				row.append((float(value), cited_ids[j]))

				if int(citing_id)!=i:
					print "Bad format in output file: '%s'!!" % output_path
					sys.exit(1)

#				if cite :
#					results[i].append((cited_ids[j]))
			
			# After all results for the given pub is are read, we sort 
			# them by value and keep only the top 100
			results[citing_ids[i]] = map(itemgetter(1), sorted(row, reverse=True)[:max])

	return results


def run(query_set, queries, ntopics) :

	data_path = DATA_PATH % query_set
	if not os.path.exists(data_path):
		print "Input folder '%s' not found. Run 'make_input_file'."
		sys.exit(1)

	exec_path = BASE_PATH + "lda"
	sets_path = BASE_PATH + "settings.txt"

	output_path = "%s/output/k%d" % (data_path, ntopics)

	if not os.path.exists(output_path) :
		os.makedirs(output_path)

	citing_tr = data_path + "citing_tr.txt"
	citing_ts = data_path + "citing_ts.txt"
	cited_ts = cited_tr = data_path + "cited.txt"

	cits_tr = data_path + "citations_tr.txt"
	cits_ts = data+path + "citations_ts.txt"

	sp.call([exec_path, "est", "0.1", sets_path, cited_tr, citing_tr, cits_tr, "seeded", output_path])

	sp.call([exec_path, "inf_new_corpus", sets_path, output_path+"/final", 
							cited_ts, citing_ts, citations_ts, output_path])

	sp.call([exec_path, "pdt", sets_path, output_path+"/final", citing_tr, 
							citations_tr, output_path+"/pdt"])


	
if __name__ == '__main__':

#	results = read_output('/home/luamct/ca/others/link_plsa_lda/output/k60/pdt-predictions.dat', 
#												'/home/luamct/ca/others/link_plsa_lda/sample_data/cited_ids.txt', 6)
#	results = read_output("/home/luamct/ca/others/link_plsa_lda/sample_data", 60, max=20)
	results = read_output("/home/luamct/data/link_plsa_lda/tunning", 60, max=20)

	
	def get_title(pid) :
		return db.select_one("title", table="papers", where="id='%s'"%str(pid))
 
	for pub, row in results.items()[:5] :
		print "%s" % get_title(pub)
		for pid in row :
			print "\t%s" % get_title(pid)
		print 

#	sys.exit()

#	set_name = 'tunning'
#	queries = load_query_set(set_name, limit=50)
#	cited_ids_map = map(itemgetter(1), 	queries)
#	make_input_file("/home/luamct/data/link_plsa_lda/" + set_name, test_ids=cited_ids_map)


#	from scipy.sparse import *
#	from scipy import *
#	row = array([0,0,1,2,2,2])
#	col = array([0,2,2,0,1,2])
#	data = array([1,2,3,4,5,6])
#	m = csr_matrix( (data,(row,col)), shape=(3,3) ).todense()
#	print m

	