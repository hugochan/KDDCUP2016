'''
Created on Jun 23, 2015

@author: luamct
'''
from evaluation.experiments import apk
import numpy as np
from ranking.searchers import Searcher
from optimizer import BayesianOptCV
from evaluation.query_sets import load_query_set
import config


class SearchEvaluator :

	def __init__(self, searcher) :
		self.searcher = searcher

	def set_params(self, **params) :
		self.searcher.set_params(**params)

	def evaluate(self, queries, metric="MAP"):
		'''
		Evaluates searcher using queries provided and the metric defined.
		Only MAP metric supported for now. 
		'''
		maps = []
		for query, pub_id, _year, actual_docs, _rels, _titles in queries :

			top = self.searcher.search(query, limit=20, exclude=set([pub_id]), force=False)
			maps.append(apk(actual_docs, top, k=20))

		return np.mean(maps)


if __name__=='__main__' :

	query_set = 'manual'
	queries = load_query_set(query_set, 30)

	se = SearchEvaluator(Searcher(**config.PARAMS))
	bocv = BayesianOptCV(estimator = se, param_bounds={
																										'K': (5, 50),
																										'papers_relev': (0.001, 1.0), 
																										'authors_relev': (0.001, 1.0), 
																										'venues_relev': (0.001, 1.0), 
																										'words_relev': (0.001, 1.0), 
																										'alpha': (0.1, 0.9),
																										'age_relev': (0.001, 1.0), 
																										'query_relev': (0.001, 1.0),
																										'ctx_relev': (0.01, 10.0)
																										 },
										 #param_list = {'kernel' : ['rbf', 'linear']},\
#										 param_fixed = {'random_state' : 1},\
										 n_jobs = 3, cv = 3, n_iter = 500,
										 gp_params = {'corr' : 'squared_exponential', 'regr' : 'constant'},
										 acq = 'ei')

	bocv.fit(queries)

