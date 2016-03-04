'''
Created on Sep 2, 2014

@author: luamct
'''
from mymysql.mymysql import MyMySQL
from collections import defaultdict, Counter
import re
import nltk
from utils import progress, tokenize
import os
import utils
import  pylucene
from config import DB_NAME, DB_USER, DB_PASSWD




def clean(s):
	return s.replace("\n", " ").strip()


def get_citing_papers(doc_id) :
	
	db = MyMySQL(db=DB_NAME, user=DB_USER, passwd=DB_PASSWD)
	
	query = """SELECT r.paper_id, 
										cg.start, cg.end 
										FROM refs r 
										JOIN citations c ON r.id=c.ref_id 
										JOIN citation_groups cg ON c.group_id=cg.id 
										WHERE cited_paper_id='%s' """ % doc_id
	rows = db.select_query(query)

	# Group citations by paper
	citations = defaultdict(list)
	for citing_paper, start, end in rows :
		citations[citing_paper].append((start, end))

	return citations


def get_cited_papers(doc_id) :

	db = MyMySQL(db=DB_NAME, user=DB_USER, passwd=DB_PASSWD)

	return db.select_query("""SELECT r.cited_paper_id, g.start, g.end 
														FROM citations c 
														JOIN citation_groups g ON c.group_id = g.id 
														JOIN refs r ON c.ref_id=r.id 
														WHERE c.paper_id='%s' AND r.cited_paper_id IS NOT NULL""" % doc_id)


def get_contexts(doc_id) :

	citations = get_citing_papers(doc_id)

	contexts = []
	for doc_id, positions in citations.items() :

		with open("/data/txt/%s.txt" % doc_id, 'r') as f :
			text = f.read()

		for (start, end) in positions :
			contexts.append( find_sentence(text, start, end) )

	return contexts


# One regex for end of sentences when going forward in the text 
# and one for when going back (reversed text).
CTX_SIZE = 300
EOS_RE_FOR = re.compile(r"\.\s+[A-Z]")
EOS_RE_REV = re.compile(r"[A-Z]\s+\.")
def find_sentence(text, s_cit, e_cit) :

	s_ctx = max(0, s_cit-CTX_SIZE)
	e_ctx = min(len(text), e_cit+CTX_SIZE)

	eos_bef = re.search(EOS_RE_REV, text[s_ctx:s_cit][::-1])
	eos_aft = re.search(EOS_RE_FOR, text[e_cit:e_ctx]) 

	start = s_ctx
	end = e_ctx
	if eos_bef :
		start = s_cit - (eos_bef.start()+1)
	if eos_aft :
		end = e_cit + (eos_aft.start()+1)

	return clean(text[start : end])


tags = None
def tag_contexts(doc_id):

	global tags
	if not tags :
		tags = nltk.data.load("help/tagsets/upenn_tagset.pickle")

	words = defaultdict(Counter)
	count = Counter()
	for context in get_contexts(doc_id) :
		for word, tag in nltk.pos_tag(tokenize(context)) :
			words[tag].update([word])

			count.update([tag])


	tag_common_words = {tag : ' '.join(zip(*tag_words.most_common(10))[0]) for tag, tag_words in words.items() }

	for tag, freq in count.most_common(15) :
		print "%4d\t%45s\t%s" % (freq, tags[tag][0], tag_common_words[tag])


def write_cited_contexts(doc_ids, index_folder, files_folder):

	contexts = defaultdict(list)
	for doc_id in progress(doc_ids):
		citations = get_cited_papers(doc_id)

		for cited, start, end in citations :
			text = utils.read_text(doc_id)

			contexts[cited].append(find_sentence(text, start, end))

# 		if len(contexts) > 100000: break

	fields = [pylucene.DocField("id", stored=True, indexed=False), 
						pylucene.DocField("contexts", stored=False, indexed=True)]
	index = pylucene.Index(index_folder, fields)


	print "Writing contexts to file for %d documents." % len(contexts)
	for i, (doc_id, ctxs) in enumerate(contexts.items()) :

		text = u"\n".join(ctxs)
		index.add(id=doc_id, contexts=text)

		# Commit and print progress every 1K entries 
		if i%1000==0 and i: 
			index.commit()
			print "%d documents indexed and written to disk." % i

		# Also write contexts into files
		with open(os.path.join(files_folder, "%s.txt"%doc_id), "w") as f :
			print >> f, text.encode("UTF-8")

	index.close()



def index_cited_contexts(doc_ids, index_folder, files_folder) :
	
	fields = [pylucene.DocField("id", stored=True, indexed=False), 
						pylucene.DocField("context", stored=False, indexed=True)]
	index = pylucene.Index(index_folder, fields)

	for i, doc_id in enumerate(doc_ids) :

		# If file doesn't exist, no context needs to be indexed
		file_path = os.path.join(files_folder, "%s.txt"%doc_id)
		if not os.path.exists(file_path) :
			continue

		# Read context text from file
		with open(file_path, "r") as f :
			contexts = unicode(f.read(), "UTF-8").split('\n')

		for context in contexts :
			index.add(id=doc_id, context=context)

		# Commit and print progress every 1K entries 
		if i%1000==0 and i: 
			index.commit()
			print "%d documents indexed." % i


def parse_context_line(line) :
	tokens_dict = {}
	doc_id, tokens_tfidf = line.strip().split('\t')
	for token_tfidf in tokens_tfidf.split():
		token,tfidf = token_tfidf.split(':')
		tokens_dict[token] = float(tfidf)
	
	return doc_id, tokens_dict


def load_contexts_tfidf(folder) :

	tokens_per_citation = {}
	for fn in progress(os.listdir(folder)) :
		fp = os.path.join(folder, fn)
		with open(fp, 'r') as f :
			for line in f :
				doc_id, tokens = parse_context_line(line)
				tokens_per_citation[doc_id] = tokens

	
	