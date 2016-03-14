'''
Created on Mar 13, 2016

@author: hugo
'''
from mymysql.mymysql import MyMySQL
import config
# from evaluation.query_sets import load_query_set
# from baselines.meng import MengSearcher
# from ranking.searchers import Searcher, PageRankSubgraphSearcher,\
#     TopCitedSubgraphSearcher, TopCitedGlobalSearcher, TFIDFSearcher, BM25Searcher,\
#     PageRankFilterBeforeSearcher, PageRankFilterAfterSearcher,\
#     GoogleScholarSearcher, ArnetMinerSearcher, CiteRankSearcher, WeightedTopCitedSubgraphSearcher
from collections import defaultdict
import time
import numpy as np

# import logging as log
import os
import cPickle
# from baselines.scholar import match_by_title
from evaluation.metrics import ndcg
#warnings.filterwarnings('error')


# log.basicConfig(format='%(asctime)s [%(levelname)s] : %(message)s', level=log.INFO)

db = MyMySQL(db=config.DB_NAME, user=config.DB_USER, passwd=config.DB_PASSWD)

def simple_selected_ranking(conf_name=None, year=None):
    """Try ranking affliations in previous years' conferences (each year each conference)
    based on how many of their papers were accepted by that conference in that year.
    """

    # Check parameter types
    if isinstance(conf_name, basestring):
        conf_name_str = "('%s')"%str(conf_name)

    elif hasattr(conf_name, '__iter__'): # length of tuple should be larger than 1, otherwise use string
        conf_name_str = str(tuple(conf_name))

    else:
        raise TypeError("Parameter 'conf_name' is of unsupported type. String or iterable needed.")

    if isinstance(year, basestring):
        year_str = "(%s)"%str(year)

    elif hasattr(year, '__iter__'): # length of tuple should be larger than 1, otherwise use string
        year_str = str(tuple(year))

    else:
        raise TypeError("Parameter 'year' is of unsupported type. String or iterable needed.")


    year_cond = "selected_papers.year IN %s"%year_str if year else ""
    conf_name_cond = "selected_papers.venue_abbr_name IN %s"%conf_name_str if conf_name else ""

    if year_cond != '' and conf_name_cond != '':
        where_cond = '%s AND %s'%(year_cond, conf_name_cond)
    elif year_cond == '' and conf_name_cond != '':
        where_cond = conf_name_cond
    elif year_cond != '' and conf_name_cond == '':
        where_cond = year_cond
    else:
        where_cond = None

    rst = db.select(['selected_papers.id', 'paper_author_affils.author_id', 'paper_author_affils.affil_id'], \
            ['selected_papers', 'paper_author_affils'], join_on=['id', 'paper_id'], \
            where=where_cond)

    # re-pack data to this format: {paper_id: {author_id:[affil_id,],},}
    pub_records = defaultdict()
    for paper_id, author_id, affil_id in rst:
        if pub_records.has_key(paper_id):
            if pub_records[paper_id].has_key(author_id):
                pub_records[paper_id][author_id].append(affil_id)
            else:
                pub_records[paper_id][author_id] = [affil_id]
        else:
            pub_records[paper_id] = {author_id: [affil_id]}

    return pub_records

def calc_ground_truth_score(pub_records): # {paper_id: {author_id:[affil_id,],},}
    affil_scores = defaultdict()
    for paper, record in pub_records.iteritems():
        score1 = 1.0 / len(record)
        for author, affils in record.iteritems():
            score2 = score1 / len(affils)
            for each in affils:
                try:
                    affil_scores[each] += score2
                except:
                    affil_scores[each] = score2


    # we only rank the selected affiliations
    selected_affils = db.select('id', 'selected_affils')
    selected_affil_scores = defaultdict()
    for each in selected_affils:
        try:
            selected_affil_scores[each] = affil_scores[each]
        except:
            pass

    selected_affil_scores  = sorted(selected_affil_scores.items(), key=lambda d: d[1], reverse=True)
    return selected_affil_scores


if __name__ == "__main__":
    import pdb;pdb.set_trace()
    pub_records = simple_selected_ranking("KDD", "2013")
    ranking = calc_ground_truth_score(pub_records)
