'''
Created on Jul 28, 2015

@author: luamct
'''
from mymysql.mymysql import MyMySQL
import config

HEADER = """\\begin{table}[!h]
\\centering
\\begin{tabular}{|c|c||c|c|}
\\hline
Node-Type & \\#Nodes & Edge-Type    & \\#Edges \\\\"""

FOOTER = """\\hline
\\end{tabular} 
\\vspace{-0.1in}
\\caption{Statistics for the employed datasets.}
\\label{tab:data}
\\end{table}
"""

TEX_NAMES = {'csx' : 'CiteSeerX',
						 'aminer': 'ArnetMiner',
						 'csx_dm': "Topic Subgraph ``data mining''",
						 'csx_ml': "Topic Subgraph ``machine learning''"}
	
	
def get_stats(dataset) :
	
	db = MyMySQL(db=dataset)

	kw_table = 'doc_ngrams' if (dataset=='aminer') else 'doc_kws'
	
	npubs = db.select_query("select count(*) from papers")[0][0]
	nauthors = db.select_query("select count(distinct author_id) from authorships")[0][0]
	nkws = db.select_query("select count(distinct ngram) from %s" % kw_table)[0][0]
	nvenues = db.select_query("select count(distinct venue_id) from papers")[0][0]

	pubs_pubs    = db.select_query("select count(*) from graph")[0][0]
	auths_auths  = db.select_query("select count(*) from coauthorships")[0][0]
	pubs_authors = db.select_query("select count(*) from authorships")[0][0]
	pubs_kws     = db.select_query("select count(*) from %s where value>=%f" % (kw_table, config.MIN_NGRAM_TFIDF))[0][0]
	
#	npubs    = 1
#	nauthors = 2
#	nkws     = 3
#	nvenues  = 4
#	pubs_pubs    = 1
#	auths_auths  = 4
#	pubs_authors = 2
#	pubs_kws     = 3

	
	print "\\hline"	
	print "\\multicolumn{4}{|c|}{%s} \\\\" % TEX_NAMES[dataset]
	print "\\hline"
	print "pubs ($N_p$) & %d & pubs-pubs & %d \\\\" % (npubs, pubs_pubs)
	print "authors   & %d & authors-authors & %d  \\\\" % (nauthors, auths_auths)
	print "keywords ($N_k$)  & %d  & pubs-keywords   & %d \\\\" % (nkws, pubs_kws)
	print "venues ($N_v$)    & %d     & pubs-authors  & %d \\\\" % (nvenues, pubs_authors)


def get_stats_table() :
	
	print HEADER
	
	datasets = ['csx', 'aminer', 'csx_dm', 'csx_ml']
	for ds in datasets :
		get_stats(ds)
	
	print FOOTER
	
	
if __name__ == '__main__':
	get_stats_table()
	
	
	