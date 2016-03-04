'''
Created on Feb 3, 2015

@author: luamct
'''

import requests
import string
import lxml.html
from pylucene import DocField, Index
from mymysql.mymysql import MyMySQL
import random
from utils import progress
import config
import os
from random import Random

#URL_TEMPLATE = "http://www.informatik.uni-trier.de/~ley/db/conf/index-%s.html"
#URL_TEMPLATE = "http://dblp.uni-trier.de/db/hc/conf/index-%s.html"
URL_TEMPLATE = "http://dblp.uni-trier.de/db/%s"

IGNORE_TERMS = ["proceedings", "proc."]

db = MyMySQL(db='aminer')


def download_venues(venue_type) :
	'''
	Venue types available: ['conf', 'journals'].
	'''
	
	folder = config.DATA + ("venues/html/%s/" % venue_type)
	url = URL_TEMPLATE % venue_type

	pos = 1
	while (True) :

		print "Processing %d-%d" % (pos, pos+99)

		resp = requests.get(url, params={'pos': pos})
		content = resp.content

		# If no results were find, time to stop
		if content.find("<em>no results</em>")>=0 :
			break

		with open(folder + ("%d.html" % pos), 'w') as file :
			print >> file, resp.content
		
		pos += 100


def remove_parenthesis(name) :
	''' Remove opening and closing parenthesis. '''
	return name.replace('(', '').replace(')', '')


def clean_journal(name) :
	''' 
	Fix names such as: "Vision; International Journal of 
	Computer ..." to "International Journal of Computer Vision".
	'''

	parts = name.split('; ')
	if len(parts)==1 :
		return clean_conf(name)

	if len(parts) > 2 :
		print "Weird format: %s" % name
		return clean_conf(name)
	
	rest, prefix = parts
	if prefix.find('...')==-1 :
		print "Weird format: %s" % name
		return clean_conf(name)
		
	fixed_name = prefix.replace(' ...', rest)
	return clean_conf(fixed_name)


def clean_conf(name) :

	if name.find(" - ")==-1 :
		return remove_parenthesis(name)
	
	abbrev, desc = name.split(' - ', 1)
	name = remove_parenthesis(desc + ' ' + abbrev)
	return name


def parse_venues(venue_types) :
	
	methods = {'conf' : clean_conf,
					 	 'journals' : clean_journal}

	venues = []
	for venue_type in venue_types :

		folder = config.DATA + ("venues/html/%s/" % venue_type)
		print "\nProcessing folder '%s'" % folder

		for file_name in os.listdir(folder) :
			print "  '%s'" % file_name
	
			with open(os.path.join(folder, file_name), 'r') as file :
				lines = file.readlines()
	
			# Get the line of interest and parse it as an HTML
			html = lxml.html.fromstring(lines[16])
	
			for item in html.xpath("//div[@id='browse-%s-output']//li/a" % venue_type) :
				process_method = methods[venue_type]

				name = process_method(item.text_content())
				venues.append((name, venue_type))


	print "%d venues." % len(venues)
	return venues


def save_venues(venues) :
	venues = [(i, name, venue_type) for i, (name, venue_type) in enumerate(venues)]
	db.insert(into="venues", fields=["id", "name", "type"], values=venues, ignore=True)


def index_venues_from_db() :

	venues = db.select(["id", "name"], table="venues")

	index = Index(config.DATA + "index_venues",
								fields=[DocField("id", stored=True, indexed=False),
												DocField("name", stored=True, indexed=True)])

	for vid, vname in venues :
		index.add(id=str(vid), name=vname)

	index.commit()
	print "%d venues added to the index." % len(venues)


	
def add_venues_to_pubs():

	index = Index(config.DATA + "index_venues")
	
#	bad_venues = db.select(fields=["papers.id", "bad_venues.name"], 
#						table=["papers", "bad_venues"], 
#						join_on=('venue_id', 'id'),
#						limit=10000)

	bad_venues = db.select(["paper_id", "venue_name"], 
												 table="temp_venues")
#	bad_venues = random.sample(bad_venues, 200)

	for pub_id, vname in progress(bad_venues):

		vname = remove_terms(vname.lower(), IGNORE_TERMS)
		cvenues, scores = index.search(vname, search_fields=["name"], 
																 return_fields=["id", "name"],
																 return_scores=True, limit=3)

		# Show the best matches
#		print "\n---------------"
#		print vname
#		for i in range(len(cvenues)) :
#			cid, cname = cvenues[i]
#			print "  [%.3f] %s" % (scores[i], cname)

		# If at least one candidate was found and the score is sufficient, update the venue 
		if (len(cvenues)>0)  and (scores[0] >= 1.0) :
			right_venue_id = int(cvenues[0][0])
			db.update(table="papers", 
								set="venue_id=%d" % right_venue_id, 
								where="id='%s'" % pub_id)

#			print "Matched!"

#			venues_mapping[vid] = 

#			abbrev, name = pubs[0]
#			if abbrev not in venues_ids :
#				venues_ids[abbrev] = (len(venues_ids), name)
#
#			good_venues.append((pub, venues_ids[abbrev][0]))

		
#		print vname

#
#	bad_venues = db.select(fields=["p.id", "v.name"], 
#						table="papers", 
#						where="venue!='' AND venue IS NOT NULL")
#
#	good_venues = []
#	venues_ids = {}
#	index = Index("eval/venues")
#	for pub, venue in progress(bad_venues):
#
#		venue = remove_terms(venue.lower(), IGNORE_TERMS)
#		docs, scores = index.search(venue, search_fields=["abbrev", "name"], 
#																 return_fields=["abbrev", "name"], 
#																 return_scores=True, limit=2)
#		if len(docs)==0 :
#			continue
#
#		if (len(scores)==1) or ((scores[0] >= 1.0) and (scores[0] >= 1.5*scores[1])) :
#
#			abbrev, name = docs[0]
#			if abbrev not in venues_ids :
#				venues_ids[abbrev] = (len(venues_ids), name)
#
#			good_venues.append((pub, venues_ids[abbrev][0]))

#		fields = index.get_documents(ids, fields=["abbrev", "name"])[0]
#		abbrev = fields['abbrev']
#		name = fields['name']
#		print venue, abbrev

	# Insert venues 
#	values = [(id, abbrev, name) for abbrev, (id, name) in venues_ids.items()]
#	db.insert(into="venues", fields=["id", "abbrev", "name"], values=values)
#
#	for pub_id, venue_id in progress(good_venues) :
#		db.update(table="papers", 
#							set="venue_id=%d"%venue_id, 
#							where="id='%s'"%pub_id)


def remove_terms(s, terms) :
	for term in terms :
		s = s.replace(term, '')
	return s


def test_random_pubs() :
	
	index = Index("eval/venues")
	
#	queries = ["BMC MEDICAL GENETICS",
#						 "PHYSICA D",
#						 "ANNUALWORKSHOP ON ECONOMICS AND INFORMATION SECURITY",
#						 "THE INTERNATIONAL JOURNAL OF ROBOTICS RESEARCH",
#						 "JOURNAL OF DISTRIBUTED AND PARALLEL DATABASES",
#						 "In Proceedings 4th Workshop on Data Mining in Bioinformatics at SIGKDD",
#						 "In Proceedings of the Twenty-First International Conference on Machine Learning"]

	pubs = db.select(["id", "title", "venue"], 
										table="papers", 
										where="(venue IS NOT NULL) AND (venue != '')", 
										limit=1000)
	pubs = random.sample(pubs, 20)

	for id, title, venue in pubs :
		venue = remove_terms(venue.lower(), IGNORE_TERMS)

		print
#		print "[Title]", title
		print "[Venue]", venue
		docs, scores = index.search(venue, 
																search_fields=["abbrev", "name"], 
																return_fields=["abbrev", "name"], 
																return_scores=True, 
																limit=3)

#		docs = index.get_documents(ids, fields=["abbrev", "name"])
		if len(scores) and scores[0]>=1.0: 
			for i in range(len(docs)) :
				abbrev, name = docs[i]
				print "  [%.3f] %s - %s" % (scores[i], abbrev, name)


def show_sample() :
	rows = db.select_query('select p.venue, concat(v.abbrev, " - ", v.name) from papers p join venues v on p.venue_id=v.id limit 50')
	for wrong, right in rows :
		print "%s\t%s" % (wrong.strip(), right.strip())


def most_popular() :
	rows = db.select_query("""select count(*) c, concat(v.abbrev, " - ", v.name) 
														from papers p join venues v on p.venue_id=v.id 
														group by p.venue_id order by c desc limit 30""")
	for count, name in rows :
		print "%s\t%s" % (count, name)


if __name__=='__main__' :

	venue_types = ['conf', 'journals']
#	download_venues('conf')
#	venues = parse_venues(venue_types)
#	save_venues(venues)
#	index_venues_from_db()
	add_venues_to_pubs()

#	venues = get_venues()
#	index_venues(venues)	
#	add_venues_to_pubs()
#	test_random_pubs()
#	show_sample()
#	most_popular()

