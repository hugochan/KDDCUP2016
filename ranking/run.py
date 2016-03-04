'''
Created on May 11, 2015

@author: luamct
'''

from mymysql.mymysql import MyMySQL
from ranking import searchers
import config
import getopt
import sys
import utils
import networkx as nx



def write_graph(graph, folder, query):
	graph_file = utils.get_graph_file_name(folder, query)
	nx.write_gexf(graph, graph_file)


def usage() :

	print "./run.py -q <query> -u usr -p pwd [-o <out_file>] [-n <n_results>] [-h]\n"
	print "      query : Query string to be searched for. Use quotes if it contains space characters."
	print "   out_file : File path of the output gexf file (default is derived the current directory)."
	print "  n_results : Maximum number of results to be returned (default is 20)."
	print "  pwd : password"
	print "  usr : username for db"
	print


def main(argv):
	query = None
	usr = None
	output_file = None
	pwd = None
	n = 20

	try:
		opts, _args_ = getopt.getopt(argv, "hq:o:n:u:p:")
	except getopt.GetoptError:
		usage()
		sys.exit(2)

	for opt, arg in opts:
			if opt == '-h':
				sys.exit()

			elif opt=="-q":
				query = arg

			elif opt=="-o":
				output_file = arg

			elif opt=="-n":
				n = int(arg)

			elif opt=="-u":
				usr = arg

			elif opt=="-p":
				pwd = arg

			else :
				print "Invalid option: %s" % opt


	# Check mandatory arguments
	if (not query or not usr or not pwd) :
		usage()
		sys.exit(2)

	s = searchers.Searcher(**config.PARAMS)
	pub_ids = s.search(query, limit=n)

	if not output_file:
		output_file = utils.get_graph_file_name(query)

	# Writes the graph structure as a gexf file
	nx.write_gexf(s.graph, output_file)

	# Prints the results
	db = MyMySQL(db='csx', user=usr, passwd=pwd)
	for id in pub_ids :
		print "%12s\t %s" % (id, db.select_one("title", table="papers", where="id='%s'"%id))


if __name__=="__main__" :
	main(sys.argv[1:])

