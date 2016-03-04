'''
Created on May 26, 2015

@author: luamct
'''
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.preprocessing.label import MultiLabelBinarizer
from mymysql.mymysql import MyMySQL
from collections import Counter, defaultdict
import numpy as np
import random
import nltk
from utils import PubTexts


db = MyMySQL(db='csx', user='root', passwd='')


#rows = db.select(fields=["id", "title", "abstract"], table="papers")
#pubs = {str(id): (title, abs) for id, title, abs in rows}
#pubs = {str(id): (title + ' ' + abs) for id, title, abs in rows}
		

MAX_KWS = 10



def get_keywords(min=1) :

	kws = db.select(fields=("id", "kw"), table=("papers", "keywords"), join_on=("id", "paper_id"))
	count = Counter([kw for _pid, kw in kws])

	unique_kws = set()
	frequent_kws = defaultdict(list)
	for pid, kw in kws :
		if (count[kw]>=min) :
			frequent_kws[pid].append(kw)
			unique_kws.add(kw)

	return frequent_kws, unique_kws


def add_words_to_db(pub_id, kws, values):
	db.insert(into="doc_kws", fields=["paper_id", "ngram", "value"], values=zip([pub_id]*len(kws), kws, values))


def get_frequent_keywords() :

	_freq_kws_, unique_kws = get_keywords(min=5)
	vocab = unique_kws - set(nltk.corpus.stopwords.words('english'))
	print "# Keywords:", len(unique_kws)

#	counter = CountVectorizer(ngram_range=(1,2), vocabulary=vocab)
	counter = TfidfVectorizer(ngram_range=(1,3), vocabulary=vocab, min_df=2, max_df=0.5)

	pubs = PubTexts()
	pub_ids = pubs.ids()
	texts = pubs.texts(pub_ids, use_title=True, use_abs=True)

	del pubs, unique_kws, _freq_kws_
	ngrams = counter.fit_transform(texts)

	print "TfIdf calculated."
	del texts

	vocab = np.asarray(counter.get_feature_names())
	for i in xrange(len(pub_ids)):
#		print texts[i]
		pub_ngrams = ngrams[i].tocoo()
		top_ngrams_idx = np.argsort(pub_ngrams.data)[::-1][:MAX_KWS]
		add_words_to_db(pub_ids[i], vocab[pub_ngrams.col[top_ngrams_idx]], pub_ngrams.data[top_ngrams_idx])

#		pub_ngrams[pub_ids[i]] = top_ngrams_idx[:MAX_KWS]
#		print "\n".join(vocab[top_ngrams_idx])
#		print


#	for pid, (title, abs) in pubs.items()[:10] :
#		ngrams = counter.fit_transform([' '.join((title, abs))])
#		top_ngrams_idx = np.argsort(ngrams.toarray()[0])[::-1]
#		pub_ngrams[pid] = top_ngrams_idx[:MAX_KWS]
#		print len(counter.get_feature_names())


#	vocab = np.asarray(counter.get_feature_names())
#	for pid, ngrams_idx in pub_ngrams.items() :
#		print pubs[pid][0]
#		print "\n".join(vocab[ngrams_idx])
#		print


#		print len(counter.vocabulary_)
#		print ngrams



def classif_missing_kws() :
	
#	texts = ["wireless networks",
#					 "networks algorithm",
#					 "algorithm em",
#					 "wireless"]
#	labels = [("l1", "l2"),
#						("l2", "l3"),
#						("l3", "l4"),
#						("l1",)]

	npubs = 10000

	kws, unique_kws = get_keywords(min=20)
	print "Total pubs:", len(kws)
	print "Unique keywords:", len(unique_kws)

#	print "\n".join(sorted(list(unique_kws)[:1000]))
#	sys.exit()

	pub_ids, labels = zip(*random.sample(kws.items(), npubs))
#	pub_ids = kws.keys()
#	labels = kws.values()
	pubs = PubTexts()
	texts = pubs.texts(pub_ids, use_title=True, use_abs=True, use_body=False)

	print "Texts loaded"
	tfidf = TfidfVectorizer(ngram_range=(1,1), stop_words="english")
	binarizer = MultiLabelBinarizer()

	x = tfidf.fit_transform(texts)
	y = binarizer.fit_transform(labels)
	print "TfIdf and labels calculated."
	del texts, labels, pub_ids

	clf = OneVsRestClassifier(LogisticRegression(), n_jobs=2)
	clf.fit(x, y)

	test = ["The Case for Wireless Overlay Networks",
					"The Cost of Adaptivity and Virtual Lanes in a Wormhole Router",
					"Robust Monte Carlo Localization for Mobile Robots",
					"Generating Finite-State Transducers For Semi-Structured Data Extraction From The Web"]
	test_x = tfidf.transform(test)
	test_y = clf.predict(test_x)
	print binarizer.inverse_transform(test_y)
	


if __name__ == '__main__':

	get_frequent_keywords()
#	classif_missing_kws()


	