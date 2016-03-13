'''
Created on Jun 9, 2015

@author: luamct
'''
from collections import defaultdict
from mymysql.mymysql import MyMySQL
from utils import progress, plot
import sys
import config
from pylucene import Index, DocField

IGNORE_TERMS = ["proceedings", "proc."]

db = MyMySQL(db='aminer', user='root', passwd='')


def load_existing_venues():
	rows = db.select(fields=["id", "name"], table="venues")
	return {name: int(id) for id, name in rows}


def save_venue(id, name) :
	db.insert(into="venues", fields=["id", "name"], values=[id, name])


def save_citations(id, cits) :
	values = [(id, cid) for cid in cits]
	db.insert(into="graph", fields=["citing", "cited"], values=values)


# Load the existing venues from the database to control the indexes
venues = load_existing_venues()

def save_pub(pub) :

	fields = []
	fields.append(('id', pub['id']))

	if 'title' in pub:
		fields.append(('title', pub['title']))

	if 'abstract' in pub:
		fields.append(('abstract', pub['abstract']))

	if ('year' in pub) and pub['year']:
		try:
			fields.append(('year', int(pub['year'])))
		# An exception will happen if year is not an integer
		except:
			pass

	if ('citations' in pub) :
		fields.append(('ncites', len(pub['citations'])))
		save_citations(pub['id'], pub['citations'])

	# Add venue
	if ('venue' in pub) :
		venue = pub['venue'].strip()
		if (venue != '') :
			if (venue not in venues) :
				venue_id = len(venues)
				venues[venue] = venue_id
				save_venue(venue_id, venue)

			fields.append(("venue_id", venues[venue]))

	name, values = zip(*fields)
	db.insert(into='papers', fields=name, values=values, ignore=True)


def read_index(line, pub) :
	pub['id'] = line

def read_title(line, pub) :
	pub['title'] = line

def read_authors(line, pub):
	pub['authors'] = line.split(';')

def read_affils(line, pub):
	pub['affils'] = line.split(';')

def read_year(line, pub):
	pub['year'] = line

def read_venue(line, pub):
	pub['venue'] = line.strip()

def read_abstract(line, pub):
	pub['abstract'] = line

def read_citations(line, pub):
	if 'citations' not in pub:
		pub['citations'] = []
	pub['citations'].append(int(line))


def import_pubs(file_path):

	handlers = {'#index' : read_index,
							'#*' : read_title,
#							'#@' : read_authors,
#							'#o' : read_affils,
							'#t' : read_year,
							'#c' : read_venue,
							'#!' : read_abstract,
							'#%' : read_citations}

	npubs = 0
	pub = {}
	file = open(file_path, 'r')
	for line in file:
		line = line.strip()

		# If record is over, save it, append it and reset the dict
		if line=='':
			save_pub(pub)
			pub = {}
			npubs += 1
			if (npubs%1000)==0:
				print "%d processed."%npubs

			continue

		try :
			id, content = line.split(' ', 1)
			if id in handlers:
				handlers[id](content.strip(), pub)

		# If an exception occurs, there was no data to be
		# processed, so just skip it
		except :
			pass


	file.close()


def read_author_name(text, record) :
	record['name'] = text

def read_affil(text, record) :
	record['affil'] = text

def read_published_count(text, record) :
	record['npubs'] = int(text)

def read_citations_count(text, record) :
	record['ncitations'] = text

def read_hindex(text, record) :
	record['hindex'] = float(text)

def read_keyterms(text, record):
	record['key_terms'] = text


def save_author(author) :

	fields, values = zip(*author.items())
	db.insert(into="authors", fields=fields, values=values, ignore=True)


def import_authors(file_path):

	handlers = {'#index' : read_index,
							'#n' : read_author_name,
							'#a' : read_affil,
							'#pc' : read_published_count,
							'#cn' :  read_citations_count,
							'#hi' : read_hindex,
							'#t' : read_keyterms}

	nauthors = 0
	author = {}
	file = open(file_path, 'r')
	for line in file:
		line = line.strip()

		# If the record is over, save it to the DB  and reset the dict
		if line=='':
			save_author(author)
			author = {}
			nauthors += 1
			if (nauthors%1000)==0:
				print "%d processed."%nauthors

			continue

		try :
			id, content = line.split(' ', 1)
			if id in handlers:
				handlers[id](content.strip(), author)

		# If an exception occurs, there was no data to be
		# processed, so just skip it
		except:
			pass

	file.close()


def import_authorships(file_path) :

	rows = []
	file = open(file_path, 'r')
	for line in progress(file) :
		_line_id, author_id, pub_id, _position = line.strip().split()
		rows.append((int(author_id), pub_id))

		# Buffer to avoid DB accesses
		if len(rows)==100:
			db.insert(into="authorships", fields=["author_id", "paper_id"], values=rows, ignore=True)
			rows[:] = []   # Empty the list

	db.insert(into="authorships", fields=["author_id", "paper_id"], values=rows, ignore=True)
	file.close()



def import_coauthorships(file_path) :

	rows = []
	file = open(file_path, 'r')
	for line in progress(file) :
		author1, author2, npubs = line[1:].strip().split()

		rows.append((int(author1), int(author2), npubs))

		# Buffer to avoid DB accesses
		if len(rows)==100:
			db.insert(into="coauthorship", fields=["author1", "author2", "npapers"], values=rows, ignore=True)
			rows[:] = []   # Empty the list

	db.insert(into="coauthorship", fields=["author1", "author2", "npapers"], values=rows, ignore=True)
	file.close()


def get_citations_cdf(citing_path, cited_path) :
	import random
	import numpy as np

	npapers = db.select_one("count(*)", table="papers")

	nciting = db.select_query("select count(*) from graph group by citing")
	nciting = (npapers-len(nciting))*[0] + [n for (n,) in nciting]
	nciting = random.sample(nciting, 100000)

	print u"%.3f \xb1 %.3f\t" % (np.mean(nciting), np.std(nciting))

	plot.cdf(nciting, title="#Citations (citing)",
					 xlabel="#Citations", ylabel="P[x $\leq$ X]",
					 xlim=(0,80), ylim=(0.5,1.0), linewidth=2,
					 outfile=citing_path)

	ncited = db.select_query("select count(*) from graph group by cited")
	ncited = (npapers-len(ncited))*[0] + [n for (n,) in ncited]
	ncited = random.sample(ncited, 100000)

	print u"%.3f \xb1 %.3f\t" % (np.mean(ncited), np.std(ncited))

	plot.cdf(ncited, title="#Citations (cited)",
					 xlabel="#Citations", ylabel="P[x $\leq$ X]",
					 xlim=(0,80), ylim=(0.5,1.0), linewidth=2,
					 outfile=cited_path)


def remove_duplicates() :

	ndels = 0

	rows = db.select_query("select title, count(*) as c from papers group by title having (c>1)")
	print "%d pubs to process" % (len(rows))

	for title, count in rows :
#		print title, count
		dups = db.select("id", table="papers", where="title='%s'"%title)

		best_id = dups[0]
		best_ncits = 0
		for dup in dups :

			ncits = db.select_one("count(*)", table="graph", where="(citing='%s' or cited='%s')" % (dup, dup))
			if (ncits>best_ncits) :
				best_id = dup
				best_ncits = ncits


		# Delete all duplicates except the best one
		for dup in dups :
			if (dup!=best_id) :
#				print "Deleting %s" % dup
				ndels += 1
				db.delete(table="papers", where="id='%s'" % dup)
				db.delete(table="graph", where="(citing='%s' or cited='%s')" % (dup, dup))

				if (ndels%1000)==0 :
					print "%d pubs deleted." % ndels


def remove_terms(s, terms) :
	for term in terms :
		s = s.replace(term, '')
	return s


def fix_venues() :
	''' Matches venues to the DBLP index so there are less dirty entries.
	'''

	index = Index(config.DATA + "index_venues",
								fields=[DocField("id", stored=True, indexed=False),
												DocField("abbrev", stored=True, indexed=True),
												DocField("name", stored=True, indexed=True)])

#	db = MyMySQL(db='aminer')
	venues = db.select(["id", "name"], table="venues")
	for _vid, vname in venues:

		vname = remove_terms(vname.lower(), IGNORE_TERMS)
		pubs, scores = index.search(vname, search_fields=["abbrev", "name"],
																 return_fields=["abbrev", "name"],
																 return_scores=True, limit=3)

		# Show the best matches
		print "\n---------------"
		print vname
#		if len(scores) and (scores[0]>=1.0):
		for i in range(len(pubs)) :
			abbrev, name = pubs[i]
			print "  [%.3f] %s - %s" % (scores[i], abbrev, name)

		if len(pubs)==0 :
			continue

		if (len(scores)==1) or ((scores[0] >= 1.0) and (scores[0] >= 1.5*scores[1])) :
			print "Matched!"

#			venues_mapping[vid] =

#			abbrev, name = pubs[0]
#			if abbrev not in venues_ids :
#				venues_ids[abbrev] = (len(venues_ids), name)
#
#			good_venues.append((pub, venues_ids[abbrev][0]))


#		print vname



if __name__ == '__main__':

	fix_venues()

#	remove_duplicates()
#	get_citations_cdf("citing.pdf", "cited.pdf")

#	import_pubs("/home/luamct/data/aminer/AMiner-Paper.txt")
#	import_authors("/home/luamct/data/aminer/AMiner-Author.txt")
#	import_authorships("/home/luamct/data/aminer/AMiner-Author2Paper.txt")
#	import_coauthorships("/home/luamct/data/aminer/AMiner-Coauthor.txt")



