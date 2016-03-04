'''
Created on Jun 23, 2015

@author: luamct
'''
import requests
from pylucene import Index
import numpy as np
import utils
from mymysql.mymysql import MyMySQL
from evaluation.query_sets import load_query_set
from utils import get_graph_file_name
import config
import os

USER_NAME = "luamct"
#URL = "http://aminer.org/services/search-publication"
URL = "https://api.aminer.org/api/search/pub"
#q=[q]&u=[u]&start=[start]&num=[num]"


def find_ids_unsupervised(titles, index_folder) :

	db = MyMySQL(db='csx')
	index = Index(index_folder)

	found = 0
	doc_ids = []
	for title in titles :
		top_docs, scores = index.search(title, 
																		search_fields=["title"], 
																		return_fields=["id"], 
																		return_scores=True,
																		limit=5)
#		ids = index.get_documents(top_docs, fields="id")

		# To decide if the most similar title in the index is a hit we check if its score 
		# is significantly higher than those of the hits that follow it (second to sixth) 

		if len(scores)>2 and (scores[0] > 2*np.mean(scores[1:])) :
			doc_ids.append(top_docs[0][0])
			found += 1
		else :
			doc_ids.append("")

		# Only enable for debugging and finding a threshold
		if 0 :
			print "-------"
			print "%s" % (title)
			print "-------"
			for i, (id,) in enumerate(top_docs) :
				title = db.select_one("title", table="papers", where="id='%s'"%id)
				print "%.2f\t%s" % (scores[i], title.encode("UTF-8"))

			if (scores[0] > 2*np.mean(scores[1:])) : 
				print "Found!", 
				op = '>'
			else : 
				print "Not found!",
				op = '<'

			print "(%.2f %s %.2f)\n" % (scores[0], op, 2*np.mean(scores[1:]))

	return doc_ids



def search_aminer(query, n) :

	start = 0

	titles = []
	while (len(titles) < 20) :
		params = {"query": query, "offset": start, "size": n, "sort": "relevance"}
		r = requests.get(URL, params=params)

		pubs = r.json()['result']
		if len(pubs)==0 :
			break

		page_titles = [pub['title'] for pub in pubs]
#		page_pub_ids = find_ids_unsupervised(page_titles, config.INDEX_PATH)

#		page_pub_ids = [pub['id'] for pub in pubs]
#		pub_ids.extend(page_pub_ids)
		titles.extend(page_titles)
		start += n

	return titles


def save_aminer_results(query_set) :

	queries = load_query_set(query_set, limit=100)
	for query, pub_id, _, _, _, _ in queries :

#		folder = config.DATA + "aminer/" + query_set

		folder = "%s/aminer/%s" % (config.DATA, config.DATASET)
		if (not os.path.exists(folder)) :
			os.makedirs(folder)

		# Only searches if file doesn't already exists
		file_path = "%s/%s.txt" % (folder, pub_id)
		if os.path.exists(file_path) :
			continue

		titles = search_aminer(query, 20)		

		print "%d\t%s" % (len(titles), query)
		with open(file_path, 'w') as file :
			print >> file, query
			for title in titles:
				print >> file, "%s" %(title.encode("UTF-8"))


if __name__ == '__main__':
	save_aminer_results("surveys")
	

#	search("subgraph mining", 20)
#	search("sentiment analysis", 20)
	
	