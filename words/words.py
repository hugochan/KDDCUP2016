'''
Created on Aug 24, 2014

@author: luamct
'''
# from topic_modeling import get_doc_ids
import os
from utils import progress
import numpy as np
from mymysql.mymysql import MyMySQL
import random
import re
from contexts import contexts
from collections import Counter, defaultdict
import nltk
import utils
from utils import tokenize
import logging
import sys
from config import DATA, TOKENS_PATH, DB_NAME
from sklearn.feature_extraction.text import TfidfVectorizer


stopwords = set(nltk.corpus.stopwords.words('english'))

NUMBER = re.compile("\d+$")
VARIABLE = re.compile("\w\d$")


db = MyMySQL(db=DB_NAME)


def filter_tokens(tokens) :
	''' 
	Filter some tokens before the analysis (infrequent, numbers, variables names).
	'''
	valid = lambda (token, _freq) : \
						not (token in stopwords) and \
						(len(token) >= 3) and \
						not re.match(NUMBER, token)

	return filter(valid, tokens)


def read_line(line) :
	token, count = line.strip().split()
	return (unicode(token, "UTF-8"), int(count))


def write_doc_vocab(ids, vocab_path) :

	print "Constructing vocabulary for %d documents." % len(ids)
	nwords = []

	# Define a quick class to use to count the tokens both 
	class Frequency:
		def __init__(self):
			self.doc = 0
			self.corpus = 0

	# Choose proper path template to get files (full_text or just important parts)

	vocab = defaultdict(Frequency)
	for i, id in enumerate(ids) :

		with open(TOKENS_PATH % id, 'r') as f :
			tokens = map(read_line, f.readlines())
			tokens = filter_tokens(tokens)

#		tokens = Counter(tokenize(text.lower()))

		# Update document frequency for each token. Note that by taking the set of 
		# the tokens above we ignore multiple occurrences within the document.
		for token, freq in tokens :
			vocab[token].doc += 1
			vocab[token].corpus += freq

		if (i%1000==0) and i :
			nwords.append((i, float(len(vocab))/1e6))
			print "%d documents loaded and vocabulary size %d." % (i, len(vocab))

	# And the vocabulary of the sample
	with open(vocab_path, 'w') as f :

		print "Writing %d words to vocabulary." % (len(vocab))
		print >> f, "%d" % len(tokens)
		for (token, freq) in vocab.items() :
			print >> f, "%s\t%d\t%d" % (token.encode("UTF-8"), freq.doc, freq.corpus)


def read_vocab(tokens_path, min_doc_freq=1, limit=100000) :
	'''
	Reads vocabulary and the document ids which created it from file.
	Also filters tokens on minimum document frequency and keeps only the
	top 'limit' most frequent tokens to control the dimensionality.
	'''

	ntokens = 0
	corpus_freqs = []
	docs_freqs = []
	tokens = []
	with open(tokens_path, 'r') as f :

		ndocs = int(f.readline().strip())
		for _lineno, line in enumerate(f.readlines()) :

			token, doc_freq, corpus_freq = line.strip().split()
			token = unicode(token ,"UTF-8")
			doc_freq, corpus_freq = int(doc_freq), int(corpus_freq) 
			ntokens += 1

			# Filter by document frequency
			if doc_freq >= min_doc_freq :			
				tokens.append(token)
				corpus_freqs.append(corpus_freq)
				docs_freqs.append(doc_freq)


	# Only keep the top 'limit' most popular 
	top_idx = np.argsort(corpus_freqs)[::-1][:limit]

	tokens = {tokens[i]: (docs_freqs[i], corpus_freqs[i]) for i in top_idx}

	logging.info("Vocabulary: %d tokens loaded, but keeping %d tokens."	% (ntokens, len(tokens)))
	return tokens, ndocs



def dump_words_tfidf(doc_ids, texts, folder, db):

	# Get vocabulary from cache (or create and dump it)
	tokens_path = os.path.join(folder, "doc_tokens.txt")
	vocab, _ndocs = read_vocab(tokens_path, min_doc_freq=10, limit=50000)

	TOP_N = 10

	words = {}
	N = float(len(doc_ids))
	for i in progress(xrange(len(texts)), 1000) :

		text = texts[i]
		doc_id = doc_ids[i]

#		for text in texts :
		tokens = Counter(tokenizer.tokenize(text)).items()

		for token, count in tokens :

			# Only include if present in vocabulary 
			if (token in vocab) :
				tf = int(count)
				idf = np.log(N/(vocab[token][0]+1.0)) + 1.0
				words[token] = tf * idf

		# Find the most relevant word according to the TF-IDf value
		top_words = sorted(words.items(), key=lambda (k,v):v, reverse=True)[:TOP_N]

		# And write them into the DB
		row_values = [(doc_id, word, tfidf) for word, tfidf in top_words]
		db.insert(into="doc_words", fields=["paper_id", "word", "value"], values=row_values)


def write_file(doc_id, tokens_tfidf) :

	with open("/data/contexts_tfidfs/%s.txt"%doc_id, "w") as f:
		for cited, ctx_tfidfs in tokens_tfidf.items() :
# 			try :
				tokens_str = ' '.join(["%s %f"%(token.encode("UTF-8"),tfidf) for token,tfidf in ctx_tfidfs])
				f.write("%s\t%s\n" % (str(cited), tokens_str))
# 			except Exception, e:
# 				import traceback
# 				print traceback.format_exc()
# 				print tokens_str


def write_contexts_vocab(ids, tokens_path) :
	'''
	Parses, tokenizes and counts the tokens in the citation contexts of all the documents.
	A document is this scenario is actually all contexts for some cited document B within 
	some document A.
	'''

	print "Constructing vocabulary for %d documents." % len(ids)

	ndocs = 0
	
	# Keep track of the document frequency and the total corpus frequency
	doc_freqs = defaultdict(int)
	corpus_freqs = defaultdict(int)
	for i, id in enumerate(ids) :

		tokens_per_citation = get_tokens_per_citation(id)
		for tokens in tokens_per_citation.values() :
			
			ndocs += 1
			for token, freq in Counter(tokens).items() :
				doc_freqs[token]    += 1
				corpus_freqs[token] += freq

		if (i%1000==0) and i :
			print "%d documents processed and vocabulary size %d." % (i, len(doc_freqs))

	
	# And the vocabulary of the sample
	with open(tokens_path, 'w') as f :

		# First line stores the total number of documents 
		# that generated this vocabulary
		print >> f, ndocs

		for token in doc_freqs.keys() :
			if doc_freqs[token] > 1 :
				print >> f, "%s\t%d\t%d" % (token.encode("UTF-8"), doc_freqs[token], corpus_freqs[token])


# def read_contexts_vocab(tokens_path) :
# 	vocab = {}
# 	with open(tokens_path, "r") as f:
# 		for line in f :
# 			token, doc_freq = line.strip().split()
# 			vocab[token] = doc_freq
# 
# 	return vocab


def get_tokens_per_citation(doc_id) :
	'''
	Fetches all cited papers by paper 'doc_id', gets the contexts around these citations and
	return them in a dict structure {cited_paper_id: [token1, token2, ..., tokenN]}. If a paper
	is cited at more than one location then the tokens for each contexts are merged together.
	'''
	citations = contexts.get_cited_papers(doc_id)

	text = utils.read_text(doc_id)

	tokens_per_citation = defaultdict(list)
	ctxs = {}
	for cited, start, end in citations :

		# Only process citation if cited paper is known (cited != None)
		if cited:
			if (start,end) not in ctxs :
				ctxs[(start,end)] = tokenizer.tokenize( contexts.find_sentence(text, start, end) )

			tokens_per_citation[cited] += ctxs[(start,end)]

	return tokens_per_citation


def get_tfidf(text, vocab, n) :
	'''
	Returns a TF-IDF representation of the given text provided the vocabulary (including 
	document frequency values) and the total number of documents.
	'''
	tfidf = {}
	tokens = tokenizer.tokenize(text)
	for token, tf in Counter(tokens).items() :

		# Only include if present in vocabulary 
		if (token in vocab) :
			idf = np.log(float(n)/(vocab[token][0]+1.0)) + 1.0
			tfidf[token] = tf*idf 

	return tfidf


def write_contexts_tfidf(path):

	# Process all available documents
	doc_ids = db.select(fields="id", table="tasks", where="status='TOKENIZED'")

	# Get vocabulary from cache (or create and dump it)
	tokens_path = path + "_tokens.txt"
	if not os.path.exists(tokens_path) :
		write_contexts_vocab(doc_ids, tokens_path)

	# Load vocabulary from file
	vocab, n = read_vocab(tokens_path, min_doc_freq=10, limit=50000)

	sys.exit()
# 	_v = sorted(vocab.items(), key=(lambda (k,(df,cf)): df))

	for doc_id in progress(doc_ids, 100) :

		# For every cited paper in doc_id we get the tokens around that citation
		tokens_per_citation = get_tokens_per_citation(doc_id)

		tokens_tfidf = defaultdict(list)
		for cited, tokens in tokens_per_citation.items() :

			counter = Counter(tokens)
			for token, tf in counter.items():

				# Only include if present in vocabulary 
				if (token in vocab) :
					idf = np.log(float(n)/(vocab[token][0]+1.0)) + 1.0
					tokens_tfidf[cited].append( (token, tf*idf) )


		# Only dump to file if there's something
		if tokens_tfidf :
			write_file(doc_id, tokens_tfidf) 

# 		print "\n".join(contexts.values()))
# 		print '\n'.join(["%15s:\t%s" % (cited, ' '.join(tokens)) for (cited, tokens) in tokens_per_citation.items()])

# 		positions = {doc_id: [(start, end) for _citing, _cited, start, end in citations]}
# 		contexts = get_contexts(positions)


def test_words(doc_ids) :
	
	n = len(doc_ids)
	while True :
		i = random.randint(0,n-1)
		doc_id = doc_ids[i]

		words = db.select(fields=["word", "value"], 
											 table="doc_words", 
											 where="paper_id='%s'"%doc_id,
											 order_by=("value","desc"),
											 limit=10)

		print "-- %s --\n" %(doc_id)
		print "\n".join([" %16s\t%.4f"%(word, tfidf) for word, tfidf in words])

		os.system("google-chrome --incognito /data/pdf/%s.pdf &> /dev/null" % doc_id)


def merge(s1, s2):
	s1 = (s1 if s1 else u'')
	s2 = (s2 if s2 else u'')
	return unicode.join(u' ', (s1, s2))


def get_value_first_sorted_row(m, i):
	vec = m.getrow(i).tocoo()
	row = [(v,c) for c,v in zip(vec.col, vec.data)]
	return sorted(row, reverse=True)


def save_frequent_ngrams_db(pubs, features, tfidf_texts):
	
	NGRAM_PER_DOC = 5

	for i in xrange(len(pubs)) :
#		print texts[i]
		row = get_value_first_sorted_row(tfidf_texts, i)

		pub_id = pubs[i][0]
		row_values = []
		for value, feat_id in row[:NGRAM_PER_DOC] :
			row_values.append( (pub_id, features[feat_id], value) )

		db.insert(into="doc_ngrams", fields=["paper_id", "ngram", "value"], values=row_values)

		if (i%1000==0) :
			print "%d processed." % i


def get_texts(start, size) :
	return db.select_query("SELECT id, title, abstract FROM papers ORDER BY id LIMIT %d OFFSET %d" % (size, start))


def get_frequent_ngrams(size=10e6) :

#	pubs = db.select(["id", "title", "abstract"], table="papers")
	start = 0
	pubs = get_texts(start, size)
	print "%s publications loaded." % len(pubs)

	texts = [merge(title, abstract) for _id_, title, abstract in pubs]
	vec = TfidfVectorizer(tokenizer=tokenize, ngram_range=(1,2), max_df=0.5, min_df=5, max_features=50000)
	tfidf_texts = vec.fit_transform(texts)

	print "TF-IDF trained."

	features = vec.get_feature_names()
	save_frequent_ngrams_db(pubs, features, tfidf_texts)
	
	# This is actually a do while statement. Check the break condition
	while (True) :
		
		start += size
		pubs = get_texts(start, size)

		# No more publications to process
		if len(pubs)==0:
			break

		texts = [merge(title, abstract) for _id_, title, abstract in pubs]
		tfidf_texts = vec.transform(texts)
		save_frequent_ngrams_db(pubs, features, tfidf_texts)



if __name__ == "__main__" :

#	nr = nc = 3
#	m = csr_matrix(np.random.uniform(size=(nr,nc)))
#	print m.todense()
#	print
#	for i in xrange(nr) :
#		print get_value_first_row(m,i)
#	sys.exit()
	
# 	folder = "cache/contexts"
# 	doc_ids = get_doc_ids(folder)
# 	test_words(doc_ids)

#	write_contexts_tfidf(DATA + "contexts_tfidfs")

#	db = MyMySQL(db="dblp")
#	ids_titles = db.select(["id", "title"], table="papers")
#	ids, titles = zip(*ids_titles)
#	dump_words_tfidf(ids, titles, "dblp/cache", db)  
	
#	print len(get_texts(200000, 10))
	get_frequent_ngrams(size=200000)


