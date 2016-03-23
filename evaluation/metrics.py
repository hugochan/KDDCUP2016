'''
Created on Jul 31, 2015

@author: luamct
'''
import numpy as np


def apk(actual, predicted, k=10):
	'''
  Computes the average precision at k. Used on MAP calculation
  (mean AP@k over multiple queries).
  '''
	if len(predicted)>k:
		predicted = predicted[:k]

	score = 0.0
	num_hits = 0.0

	for i,p in enumerate(predicted):
		if (p in actual) and (p not in predicted[:i]):
			num_hits += 1.0
			score += num_hits / (i+1.0)

#		if not actual:
#			return 0.0
	return score / min(len(actual), k)


def ndcg2(actual, pred, relevs=None, k=20):
	''' Normalized Discounted Cummulative Gain. '''

	if not relevs :
		relevs = [2.0]*len(actual)

	pred = pred[:k]
	relevs_dict = {actual[i]: relevs[i] for i in xrange(len(actual))}

	r = [relevs_dict[item] if item in relevs_dict else 0.0 for item in pred]

	ideal_r = sorted([relevs_dict[item] for item in actual], reverse=True)[:k]

	idcg = dcg(ideal_r)
	return dcg(r)/idcg if idcg!=0.0 else 0.0


def ndcg(actual, pred, relevs=None, k=20):
	''' Normalized Discounted Cummulative Gain. '''

	if not relevs :
		relevs = ["R1"]*len(actual)

	pred = pred[:k]
	relevs_values = {"R1":2.0, "R2":1.0}
	relevs_dict = {actual[i]: relevs_values[relevs[i]] for i in xrange(len(actual))}

	r = [relevs_dict[doc_id] if doc_id in relevs_dict else 0.0 for doc_id in pred]
#	ideal_r = sorted(r, reverse=True)
	ideal_r = sorted([relevs_dict[doc_id] for doc_id in actual], reverse=True)[:k]

	idcg = dcg(ideal_r)
	return dcg(r)/idcg if idcg!=0.0 else 0.0


def dcg(relevs):
	''' Discounted Cummulative Gain. '''

	if len(relevs) == 0 :
		return 0.0

	v = relevs[0]
	for i in xrange(1, len(relevs)) :
		v += relevs[i]/np.log2(i+1)

	return v


def recall_at(actual, pred, k=20) :
	''' Recall at the top k values. '''
	pred = set(pred[:k])
	actual = set(actual)

	return float(len(actual & pred))/len(actual)


def precision_at(actual, pred, k=20):
	''' Precision at the top k values. '''
	pred = set(pred[:k])
	actual = set(actual)

	return float(len(actual & pred))/k
