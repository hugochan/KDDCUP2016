'''
Created on Jun 8, 2014

@author: luamct
'''

import os
import re
from _collections import defaultdict
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import Counter
from itertools import izip
from scipy.spatial.distance import cosine
#from utils import plot  # @UnresolvedImport
import random
import matplotlib.pyplot as pp


nltk.data.path.append('/home/luamct/.nltk_data/')
tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")


class SectionNotFound(Exception) :
	''' Raised when the section is not find the text. '''
	pass


def get_section(text, section_begin, section_end):
	
	beg_regex = re.compile("|".join(section_begin), re.IGNORECASE)
	end_regex = re.compile("|".join(section_end), re.IGNORECASE)

	begs = list(re.finditer(beg_regex, text))

	if not begs :
		raise SectionNotFound()
	
	# Get last occurrence of the regex as the beginning of the section
	begin = begs[-1].start()
	
	# Search for possible end points for the section
	ends = list(re.finditer(end_regex, text[begin:]))

	# Get the first candidate for references ending. If 
	# none found, just use None for the end of the string.
	if not ends :
		raise SectionNotFound()

	end = ends[0].start()
	
	return text[begin:begin+end]



def get_citation_positions(db, paper_id) :
	query = """SELECT r.paper_id, 
										cg.start, cg.end 
										FROM refs r 
										JOIN citations c ON r.id=c.ref_id 
										JOIN citation_groups cg ON c.group_id=cg.id 
										WHERE cited_paper_id='%s' """ % paper_id
	cursor = db.query(query)
	rows = cursor.fetchall()

	# Group citations by paper
	citations = defaultdict(list)
	for citing_paper, start, end in rows :
		citations[citing_paper].append((start, end))

	return citations


def sentence_limit(text, start, direction):
	
	if direction==1 :
		end = len(text)
	elif direction==-1:
		end = 0
	else :
		raise TypeError("Unsupported direction value passed: %d." % direction)

	curr = start+direction
	next = curr+direction
	while (curr!=end) :

		if text[curr] == '.' :
			break

		curr += direction
	
	return curr+1


def clean(s):
	return s.replace("\n", "")


def citation_contexts(citations_positions) :

	sentences = []
	for paper_id, positions in citations_positions.items() :

		txt_file = "/data/txt/%s.txt" % paper_id

		with open(txt_file, 'r') as f :
			text = f.read()
			
		for (start, end) in positions :
			ss = sentence_limit(text, start, -1)
			se = sentence_limit(text, end-1, 1)
			sentences.append( clean(text[ss:start] + text[end:se]) )
			
	return sentences
			

def get_cited(db, limit=100):
	
	query = """select distinct cited_paper_id 
						 from refs r join tasks t on r.cited_paper_id = t.id 
						 where cited_paper_id is not NULL
						 and t.status = 'CONVERTED'
						 limit %s""" % limit
	c = db.query(query)
	cited = c.fetchall()

	# Detuple before returning
	return [c for (c,) in cited]


def remove_letters(words):
	not_letter = lambda (w, c) : len(w)>1
	return filter(not_letter, words)


def make_word_cloud(text, filepath):
	import wordcloud  #@UnresolvedImport

	if isinstance(text, str) :
		text = wordcloud.process_text(text, max_features=20)
		
	w, h = (400, 400)
	text = remove_letters(text)
	elements = wordcloud.fit_words(text, width=w, height=h)

	wordcloud.draw(elements, filepath, width=w, height=h, scale=1)
	return filepath


def progress(items, checkpoint=1000) :

	for i, item in enumerate(items):
		if i and (i%checkpoint==0) :
			print "%d processed." % i
		yield item
		
		
def merge_images(path1, path2, outpath) :

	from PIL import Image   # @UnresolvedImport
	i1 = Image.open(path1)
	i2 = Image.open(path2)
	
	w1, h1 = i1.size
	w2, h2 = i2.size  # @UnusedVariable
	
	merged = Image.new("RGBA", (w1+w2, h1))
	merged.paste(i1, (0,0))
	merged.paste(i2, (w1,0))
	
	merged.save(outpath)


def get_common(i, terms, freqs, ntop) :
	
	occurences = Counter()
	row = freqs.getrow(i).tocoo()
	for j, v in zip(row.col, row.data) :
		occurences[terms[j]] = v

	return occurences.most_common(ntop)
	

def single_word_cloud(i, pid, terms1, terms2, freqs1, freqs2):

	common1 = get_common(i, terms1, freqs1, 20)
	common2 = get_common(i, terms2, freqs2, 20)

	img1 = make_word_cloud(common1, "/tmp/1.png")
	img2 = make_word_cloud(common2, "/tmp/2.png")

	img_file = "clouds/%s.png"%pid[i]
	merge_images(img1, img2, img_file)
	pp.imshow(pp.imread(img_file))
	pp.show()
# 	os.system("gnome-open clouds/%s.png &> /dev/null" % ids[i])
	

def to_dict(terms, freqs):
	''' 
	Represents the frequency array as a dictionary {term: freq}. 
	'''
	return {term: freq for (term, freq) in izip(terms, freqs)}


def to_same_dimension(terms, fmap1, fmap2):

	nterms = len(terms)
	vec1 = np.zeros(nterms, np.float)
	vec2 = np.zeros(nterms, np.float)

	for i, term in enumerate(terms) :

		if term in fmap1 : vec1[i] = fmap1[term]
		if term in fmap2 : vec2[i] = fmap2[term]

	return vec1, vec2


def get_texts(db, limit=1000):
	paper_ids = get_cited(db, limit)

	ignored = 0

	ids = []
	important_texts = []
	citations_texts = []
	for paper_id in progress(paper_ids) :

		# Read the file's content
		txt_file  = os.path.join("/data/txt/%s.txt" % paper_id)
		with open(txt_file, 'r') as f :
			text = f.read()

		try :
			# Try to find the abstract and conclusion
			abstract = get_section(text, ["abstract"], ["introduction"])
			conclusion = get_section(text, ["conclusion"], ['references', 'bibliography', 'acknowledg', 'appendix'])
			important = abstract.strip() + conclusion.strip()

			# Get the text around the citations of the current paper
			citations_pos = get_citation_positions(db, paper_id)
			contexts = citation_contexts(citations_pos)
			contexts = " ".join(contexts).strip()

			# If successful so far, add to the result lists			
			ids.append(paper_id)
			important_texts.append(important)
			citations_texts.append(contexts)

		except SectionNotFound:
			ignored += 1

	print "%d documents ignored." % ignored

	return ids, important_texts, citations_texts


def texts_tfidf(ids, important_texts, citations_texts) :
	'''
	Generates tf-idf vectors for each text then calculates cosine similarity between the vectors. 
	'''

	tfidf = TfidfVectorizer(strip_accents='ascii',
													stop_words='english', 
													ngram_range=(1,2),
													min_df=2)

	freqs1 = tfidf.fit_transform(important_texts)
	terms1 = tfidf.get_feature_names()

	freqs2 = tfidf.fit_transform(citations_texts)
	terms2 = tfidf.get_feature_names()

	return terms1, terms2, freqs1, freqs2


def texts_similarity(terms1, terms2, freqs1, freqs2) :

	# Merge all terms
	terms = list(set(terms1 + terms2))

	npapers = freqs1.shape[0]
	sims = np.empty(npapers, np.float)
	
	for i in xrange(npapers) :

		# If one of the vectors is nil, skip it
		if (freqs1[i].sum()==0.0) or (freqs2[i].sum()==0.0) :
			continue

		# Changes representation to a {term: freq} map
		fmap1 = to_dict(terms1, freqs1.getrow(i).toarray()[0])
		fmap2 = to_dict(terms2, freqs2.getrow(i).toarray()[0])

		vec1, vec2 = to_same_dimension(terms, fmap1, fmap2)

		sims[i] = 1.0-cosine(vec1, vec2)

	return sims


def random_similarity(terms1, terms2, freqs1, freqs2) :

	# Merge all terms
	terms = list(set(terms1 + terms2))

	npapers = freqs1.shape[0]
	sims = np.empty(npapers, np.float)

	for i in xrange(npapers) :
		a = random.randint(0,npapers-1)  #@UndefinedVariable
		b = random.randint(0,npapers-1)	 #@UndefinedVariable

		# If one of the vectors is nil, skip it
		if (freqs1[a].sum()==0.0) or (freqs2[b].sum()==0.0) :
			continue

		# Changes representation to a {term: freq} map
		fmap1 = to_dict(terms1, freqs1[a].toarray()[0])
		fmap2 = to_dict(terms2, freqs2[b].toarray()[0])

		vec1, vec2 = to_same_dimension(terms, fmap1, fmap2)

		sims[i] = 1.0-cosine(vec1, vec2)

	return sims


def main()  :
	pass

# 	db = MySQL(host="localhost", user="root", passwd="", db="csx")
# 
# 	ids, important_texts, citations_texts = get_texts(db, limit=1000)
# 	terms1, terms2, freqs1, freqs2 = texts_tfidf(ids, important_texts, citations_texts)
# 
# 	same_sims = texts_similarity(terms1, terms2, freqs1, freqs2)
# 	rand_sims = random_similarity(terms1, terms2, freqs1, freqs2)
# 
# 	sorted_sims = np.argsort(same_sims)

# 	for pid in ['10.1.1.157.9878', '10.1.1.158.356'] :
# 	for pid in ['10.1.1.157.7093', '10.1.1.157.519', '10.1.1.158.5759'] :
# 		single_word_cloud(ids.index(pid), ids, terms1, terms2, freqs1, freqs2)

# 	while True :
# 		i = random.randint(0, len(ids)-1)
# 		print ids[i]
# 		single_word_cloud(i, ids, terms1, terms2, freqs1, freqs2)

# 	for high in sorted_sims[-10:] :
# 		print ids[high]
# 		single_word_cloud(high, ids, terms1, terms2, freqs1, freqs2)
# 
# 	for low in sorted_sims[:10] :
# 		print ids[low]		
# 		single_word_cloud(low, ids, terms1, terms2, freqs1, freqs2)
						
	# Dump to file so no need for re-computing when re-displaying
# 	pickle.dump((same_sims, rand_sims), open('data/sims.p', 'w'))

#	(same_sims, rand_sims) = pickle.load(open('data/sims.p', 'r'))
#	plot.histogram(same_sims, outfile='sim_hist.pdf')
# 	plot.histogram(rand_sims)
# 	plot.histogram((same_sims, rand_sims))
	
	
if __name__ == '__main__':
	main()
	
	

		

