'''
Created on May 14, 2015

@author: luamct
'''
import config
import os
import random
from mymysql.mymysql import MyMySQL
import utils
import nltk
from pylucene import Index
import string
import re


# Stop words to be removed from the title to make a query
_stop_words_ = set(nltk.corpus.stopwords.words('english'))

# Regex to remove some punctuation effectively 
_regex_ = re.compile('[%s]' % re.escape("!\"#$%&'()*+,.:;?`{|}~"))


def load_query_set(query_set, limit=100):

	folder_path = config.QUERY_SETS_PATH + query_set
	file_path = folder_path + ".txt"

#	file_names = os.listdir(folder)
#	limit = min(len(file_names), limit)

	# Load query file
	queries = []
	with open(file_path) as file : 
		for line in file :
			queries.append(line.strip().split('\t'))

	queries = queries[:limit]

	#	if limit and (limit<len(queries)) :
	#		queries = random.sample(queries, limit)


	queries_cits = []
	for pub_id, year, _title_, query in queries:

		# Query is the file name, excluding the suffix after the char _ (numeric 
		# identifier for multiple identical queries) and replacing + by an space char
#		query = fn[:-4].split("_")[0]     
#		query = query.replace('+', ' ')

		relevs = []
		doc_ids = []
		titles = []
		cits_file_path = "%s/%s.txt" % (folder_path, pub_id)
		with open(cits_file_path, 'r') as file :

			# Ignore the header
			_header_ = file.readline()
#			print fn, header
#			query_doc_id, year = header.strip('\n').split('\t')

			for line in file :
				relev, cits_id, title = line.strip().split('\t')

				if (cits_id != "") :
					relevs.append(relev)
					doc_ids.append(cits_id)
					titles.append(unicode(title, "UTF-8"))

			queries_cits.append((unicode(query, "UTF-8"), pub_id, year, doc_ids, relevs, titles))

	return queries_cits


def has_similar_pub(db, index, pub_id, title, citations) :

	# Fetches all document that have at least one of the terms
	similar_pubs = index.search(title.strip(),
													 search_fields=["title", "abstract"], 
													 return_fields=["id", "title"],
													 ignore=set([pub_id]), limit=1)

	similar_citations = utils.get_cited(db, similar_pubs[0][0])

#		print "\n-------"
#		print "%s\t%s" % (pub_id, title)
#		print "-------"
#		for sp_id, sp_title in similar_pubs :
#			sp_citations = utils.get_cited(db, sp_id)
#			in_common = len(set(citations) & set(sp_citations))
#			print "%d\t%15s\t%s" % (in_common, sp_id, sp_title)

	n_cits = float(len(similar_citations))
	common = len(set(similar_citations) & set(citations))
	return (n_cits != 0) and (common/n_cits > 0.5)


def to_query(title) :
	'''
	Removes stop words and punctuations. Uses global variables!
	'''
	title = _regex_.sub('', title)
	title_words = title.strip().lower().split()
	query_words = [word for word in title_words if word not in _stop_words_]
	return ' '.join(query_words)


def save_to_file(prefix, pubs):
	'''
	Takes the list of pubs (id, title, query, year) and 
	dumps into the given file path.
	'''
	file_path = prefix + ".txt"
	with open(file_path, 'w') as file :
		for pub_id, title, query, year in pubs :
			print >> file, "%s\t%d\t%s\t%s" % (pub_id, year, 
																				title.encode("UTF-8"), 
																				query.encode("UTF-8"))


def write_citations_query_set_files(db, prefix1, n1, prefix2, n2) :
	'''
	Sample random papers meeting some criteria to be used as ground truth 
	(title is used as query and the citations as the expected list of 
	relevant papers).

	Two non overlapping sets are created to be used as tuning and testing.
	'''

	# The index is used to find very similar publications
	index = Index(config.INDEX_PATH)
	index.attach_thread()

#	random.seed(86)  #@UndefinedVariable
	docs = db.select(["id", "title", "year"], 
									 table="papers", 
									 where="use_it AND (year IS NOT NULL) AND (year != 0)")

	sample = []
	while (len(sample) < (n1+n2)) :

		pub_id, title, year = random.choice(docs)  #@UndefinedVariable
		title = title.strip()

		citations = utils.get_cited(db, pub_id)
		if (len(citations) >= 20) :

			if not has_similar_pub(db, index, pub_id, title, citations) :
				query = to_query(title)
				sample.append((pub_id, title, query, year))
				print len(sample)
			else :
				print "Ignoring: \t'%s'" % title

	# Shuffle before splitting the sets into tuning and testing
	random.shuffle(sample)  #@UndefinedVariable
	set1 = sample[:n1]
	set2 = sample[n1:]

	save_to_file(prefix1, set1)
	save_to_file(prefix2, set2)


def write_query_set_folder(db, prefix) :
	'''
	Load queries from the prefix.txt, get citations for them
	and write each to a single file under folder prefix
	'''
	# Create folder if it doesn't exist
	if not os.path.exists(prefix) :
		os.mkdir(prefix)

	queries_file_path = prefix + ".txt"
	with open(queries_file_path, 'r') as file :

		for line in file :
			pub_id, year, title, _query_ = line.strip().split('\t')


			file_path = "%s/%s.txt" % (prefix, pub_id)

			citations = db.select("cited", table="graph", where="citing='%s'"%pub_id)

			# Write seed document id and then one citation per line
			with open(file_path, 'w') as citations_file :
				print >> citations_file, "%s\t%s\t%s" % (pub_id, year, title)
				for cited in citations:
					title = utils.get_title(db, cited).strip()
					print >> citations_file, "%s\t%s\t%s" % ("R1", cited, title.encode("UTF-8"))


def write_citations_queries(name1, n1, name2, n2) :

	db = MyMySQL(db=config.DB_NAME)

	if not os.path.exists(config.QUERY_SETS_PATH) :
		os.mkdir(config.QUERY_SETS_PATH)

	path1 = config.QUERY_SETS_PATH + name1
	path2 = config.QUERY_SETS_PATH + name2

#	write_citations_query_set_files(db, path1, n1, path2, n2)

	write_query_set_folder(db, path1)
	write_query_set_folder(db, path2)


def check_overlap(query_set1, query_set2) :

	def read_ids(file_path) :
		ids = []
		with open(file_path, 'r') as file :
			for line in file :
				ids.append(str(line.split('\t')[0]))
		return ids

	file1 = config.QUERY_SETS_PATH + query_set1 + ".txt"
	file2 = config.QUERY_SETS_PATH + query_set2 + ".txt"

	set1 = set(read_ids(file1))
	set2 = set(read_ids(file2))

	print len(set1), len(set2), len(set1 & set2)


def write_surveys_queries_file(prefix, npubs=110) :

	db = MyMySQL(db=config.DB_NAME)
	candidates = db.select_query('''SELECT id, substring(title,1,140), year 
																	FROM papers 
																	WHERE title LIKE '%survey%' AND (year IS NOT NULL)
																	AND (year BETWEEN 1950 AND 2014)''')

	print "Candidates: %s" % len(candidates)

	# Include the word 'survey' for this particular case
	_stop_words_.add("survey")

	# Write candidates to file	
	file = open(prefix + ".txt", "w")

	n = 0
	for pub_id, title, year in candidates :

		citations = utils.get_cited(db, pub_id)
		if len(citations)>=20 :
			query = to_query(title)

			print >> file, "%s\t%d\t%s\t%s" % (pub_id, year, title.strip(), query)

			n += 1
			if (n >= npubs) :
				break

	file.close()


def write_surveys_queries(n=110) :
	
	db = MyMySQL(db=config.DB_NAME)

	if not os.path.exists(config.QUERY_SETS_PATH) :
		os.mkdir(config.QUERY_SETS_PATH)

	prefix = config.QUERY_SETS_PATH + "surveys"

#	write_surveys_queries_file(prefix, n)
	write_query_set_folder(db, prefix)


def check_ids(folder) :

	db = MyMySQL(db='csx')

	for i in xrange(1,8) :
		print i
		print

		with open(folder + str(i) + ".txt") as file :

			_header_ = file.readline()
			for line in file :

				relev, pub_id, title = line.strip().split('\t')
				if (len(db.select("id", table="papers", where="id='%s'"%pub_id)) == 0) : 
					print "Pub not found:", pub_id

#				print pub_id, year, title


def match_pubs(index, raw_file_path, matched_file_path) :
	
	pubs = []

	# Opens raw file and skips first line
	in_file = open(raw_file_path, 'r')

	query = in_file.readline().strip()
	print '\n-- %s --' % query

	for line in in_file :
		relev, title = line.strip().split('\t')

		# Fetches all document that have at least one of the terms
		candidates = index.search(title.strip(),
														 search_fields=["title"], 
														 return_fields=["id", "title"], 
														 limit=1)

		cand_id, cand_title = candidates[0]
		
		title = title.strip()
		ctitle = cand_title.strip().encode("UTF-8")

		print "\n-- %s" % title
		print "   %s" % ctitle

#		for cand_id, ctitle in candidates :
#			print "  %s" % ctitle.encode("UTF-8")

		if (title.lower()==ctitle.lower()) :
			print 'Matched!'

		else :
			matched = raw_input("Matched? ")
			if (matched!='y') :
				cand_id = ''
	
			if matched=='q' :
				break

		# Id will be empty if not matched, otherwise will be the matched id
		pubs.append((relev, cand_id, title))

	in_file.close()

	# Now write down the file for this query. 
	# Unmatched pubs will have empty ids
	print "Saving '%s'." % matched_file_path
	with open(matched_file_path, 'w') as out_file :

		print >> out_file, "\t\t%s" % query
		for relev, pub_id, title in pubs :
#			try :
				print >> out_file, "\t".join([relev, pub_id, title.encode("UTF-8")])
#			except Exception, e:
#				print e


def write_manual_queries() :

	raw_folder = config.DATA + "manual_raw"
	matched_folder = config.QUERY_SETS_PATH + "manual/"
	
	# The index is used to find very similar publications
	index = Index(config.INDEX_PATH)
	index.attach_thread()

	# Create folder if it doesn't exist
	if not os.path.exists(matched_folder) :
		os.mkdir(matched_folder)

	file_names = sorted(os.listdir(raw_folder))[:3]
#	file_names = ['9.txt', '10.txt']

	for file_name in file_names :
#		print '\n-- %s --\n' % file_name

		raw_file_path = os.path.join(raw_folder, file_name)
		matched_file_path = os.path.join(matched_folder, file_name)
		match_pubs(index, raw_file_path, matched_file_path)



def manual_queries_topic_graphs(from_dataset, to_dataset) :

	db = MyMySQL(db=to_dataset)
	pub_ids = set(db.select("id", table="papers"))

	from_folder = config.DATA + "query_sets/" + from_dataset + "/manual/"
	to_folder = config.DATA + "query_sets/" + to_dataset + "/manual/"

	for file_name in os.listdir(from_folder) :

		print file_name
		from_file = open(from_folder + file_name, 'r')
		to_file = open(to_folder + file_name, 'w')

		# Read and write back header line
		header = from_file.readline().strip('\n') # ignore header
		print >> to_file, header

		for line in from_file :
			relev, pub_id, title = line.strip().split('\t')
			if (pub_id not in pub_ids) :
				pub_id = ''

			print >> to_file, "%s\t%s\t%s" %(relev, pub_id, title)

		from_file.close()
		to_file.close()


if __name__ == '__main__' :
	pass
#	write_manual_queries()

#	manual_queries_topic_graphs("csx", "csx_dm")
#	manual_queries_topic_graphs("csx", "csx_ml")

#	check_ids(config.QUERY_SETS_PATH + "manual/")

#	write_surveys_queries(n=33)
#	write_citations_queries("tuning", 32, "testing", 62)

#	check_overlap('citations', 'tuning')

