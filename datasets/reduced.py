'''
Created on Jun 29, 2015

@author: luamct
'''
from mymysql.mymysql import MyMySQL
import random
import networkx as nx
from utils import progress


db = MyMySQL(db='csx')


def get_cited(pub_id):
	return db.select("cited", table="graph", where="citing='%s'"%pub_id)

def get_citing(pub_id):
	return db.select("citing", table="graph", where="cited='%s'"%pub_id)

def get_neighbours(pub_id):
	return db.select(["citing", "cited"], table="graph", where="citing='%s' OR cited='%s'" % (pub_id, pub_id))


def depth_walk() :
	
	ids = db.select("id", table="papers", limit=10000)

	queue = []
	visited = set()

	pub_id = random.choice(ids)  #@UndefinedVariable
	queue.append(pub_id)
	print pub_id

	while len(queue) :
		print "Queue size: %d" % len(queue)

		next_id = queue.pop(0)
		if next_id not in visited :
			cits = get_cited(next_id)
			queue.extend(cits)
			visited.add(next_id)

		if len(visited) > 50000 :
			print "Too many visited!"
			break

	print cits
	

def top_centrality() :
	
	citations = db.select(["citing", "cited"], table="graph", limit=1000000)
	print len(citations)

	graph = nx.DiGraph()
	for citing, cited in progress(citations, 10000) :
		graph.add_edge(int(citing), int(cited))

	print graph.number_of_nodes()
	centrality = nx.betweenness_centrality(graph)
	print centrality.items()[:100]
	

def get_next_hop(nodes) :
	
	next_nodes = set()
	for node in progress(nodes, 100) :
		next_nodes.update(get_citing(node))
		next_nodes.update(get_cited(node))

	return next_nodes


def keyword_centric(keyword, from_db, to_db) :

	db = MyMySQL(db=from_db)
	pub_ids = db.select("paper_id", table="keywords", where="kw='%s'"%keyword)

	nodes = set()
	new_nodes = set()
	new_nodes.update(pub_ids)

	n = 50000
	while len(nodes) < n :

		new_nodes = get_next_hop(new_nodes)
		nodes.update(new_nodes)
		print len(nodes)


	print "Adding %d nodes." % len(nodes)

	new_db = MyMySQL(db=to_db)

#	values = ','.join(['%s'%id for id in nodes])
	new_db.insert(into="use_papers", fields=["paper_id"], values=list(nodes))


if __name__ == '__main__':
#	depth_walk()
#	top_centrality()

	keyword_centric("data mining", from_db="csx", to_db="csx_dm")
#	keyword_centric("machine learning", from_db="csx", to_db="csx_ml")
	
	

