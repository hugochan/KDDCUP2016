'''
Created on Mar 14, 2016

@author: hugo
'''

import os
import sys
import json
import time
import numpy as np
import config
from collections import defaultdict
from mymysql.mymysql import MyMySQL
from ranking.kddcup_searchers import simple_search, SimpleSearcher, RegressionSearcher, \
                    Searcher, ProjectedSearcher, IterProjectedSearcher, StatSearcher, \
                    TemporalSearcher, Bagging
from evaluation.kddcup_expt import get_results_file, save_results

db = MyMySQL(db=config.DB_NAME, user=config.DB_USER, passwd=config.DB_PASSWD)



def rank_affils(selected_affils, conf_name, year, searcher, show=True, results_file=None, bagging_list=[]):
    conf_id = db.select("id", "confs", where="abbr_name='%s'"%conf_name, limit=1)[0]
    start = time.time()

    if searcher.name() == "SimpleSearcher":
        expand_year = []
        # expand_year = range(2005, 2011)
        results = searcher.search(selected_affils, conf_name, year, expand_year=expand_year, age_decay=True, rtype="affil")
        # # normalize ranking score
        # ids, scores = zip(*results)
        # scores = np.array(scores)
        # _max = np.max(scores)
        # if _max !=  0:
        #     scores = scores / _max
        # results = zip(ids, scores.tolist())


    elif searcher.name() == "IterProjectedLayered":
        expand_year = []
        results = searcher.easy_search(selected_affils, conf_name, year, [], expand_year, [])

    elif searcher.name() == "StatSearcher":
        expand_year = []
        # expand_year = range(2005, 2011)

        results = searcher.search(selected_affils, conf_name, year, [], expand_year, force=True, rtype="affil")


    elif searcher.name() == "TemporalSearcher":
        expand_year = []
        # expand_year = range(2005, 2011)
        # expand_conf_year = [("ICDM", range(2011, 2014))]
        expand_conf_year = []

        # results = searcher.author_search(selected_affils, conf_name, year, exclude_papers, expand_year, force=True, rtype="affil")
        results = searcher.affil_search(selected_affils, conf_name, year, [], expand_year, expand_conf_year=[], force=True, rtype="affil")

    elif searcher.name() == "MultiLayered":
        expand_year = []
        # expand_conf_year = [("ICDM", range(2011, 2014))]
        expand_conf_year = []

        results = searcher.search(selected_affils, conf_name, year, [], expand_year, expand_conf_year, force=True, rtype="affil")

    elif searcher.name() == "Bagging":
        results = searcher.bagging(conf_name, bagging_list)


    else:
        return []


    print "Runtime: %s" % (time.time() - start)

    # normalize ranking score
    ids, scores = zip(*results)
    scores = np.array(scores)
    _max = np.max(scores)
    if _max !=  0:
        scores = scores / _max
    results = zip(ids, scores.tolist())

    pred = zip(*results)[0]

    # write to disk
    with open("rst_%s_%s.json"%(conf_name, searcher.name()), 'w') as f:
        json.dump(dict(zip(pred, range(len(pred)))), f)
        f.close()


    with open("rst_%s_%s_score.json"%(conf_name, searcher.name()), 'w') as f:
        json.dump(dict(results), f)
        f.close()


    if results_file:
        save_results(conf_id, results, results_file)



def merge_confs_in_phase(phase, method_name):
    confs = [
                [
                    # "SIGIR", # Phase 1
                    # "SIGMOD",
                    # "SIGCOMM"
                ],

                [
                    "KDD", # Phase 2
                    "ICML"
                ],

                [
                    # "FSE", # Phase 3
                    # "MobiCom",
                    # "MM"
                ]
            ]

    if not phase in range(1, 4):
        return
    rfile_list = []
    for c in confs[phase - 1]:
        rfile_list.append(get_results_file(c, "results_%s"%method_name))

    folder = "%sresults/phase_%s" % (config.DATA, phase)
    if not os.path.exists(folder) :
        os.makedirs(folder)

    print "Merge result files ..."
    try:
        with open("%s/results.tsv"%folder, 'w+') as outfile:
            for fname in rfile_list:
                try:
                    with open(fname, 'r') as infile:
                        for line in infile:
                            outfile.write(line)
                except Exception, e:
                    print e
                    continue
                infile.close()
    except Exception, e:
        print e

    outfile.close()
    print "Generate final submission file: %s/results.tsv"%folder


def main():

    confs = [
                # "SIGIR", # Phase 1
                # "SIGMOD",
                # "SIGCOMM",

                "KDD", # Phase 2
                "ICML",

                # "FSE", # Phase 3
                # "MobiCom",
                # "MM",
            ]

    searchers = [
                    # SimpleSearcher(**config.PARAMS),
                    # # RegressionSearcher(**config.PARAMS),
                    # Searcher(**config.PARAMS),
                    # # ProjectedSearcher(**config.PARAMS),
                    # IterProjectedSearcher(**config.PARAMS),
                    # StatSearcher(**config.PARAMS),
                    # TemporalSearcher(**config.PARAMS),
                ]

    # import pdb;pdb.set_trace()
    selected_affils = db.select(fields="id", table="selected_affils")
    year = ["2011", "2012", "2013", "2014", "2015"]

    bagging_list = ["SimpleSearcher", "MultiLayered", "IterProjectedLayered"]

    for c in confs:
        print "Running on '%s' conf." % c

        for s in searchers:
            print "Running %s." % s.name()

            if s.name() == "SimpleSearcher":
                s.set_params(**{
                              'age_relev': .2, # .5, .7, .08, .2
                              })

            if s.name() == "RegressionSearcher":
                s.set_params(**{
                              'age_relev': .0, # .5, .7, .08
                              })

            if s.name() == "MultiLayered":
                s.set_params(**{
                              'H': 0,
                              'age_relev': 0.1, # 0.01
                              'papers_relev': .99, # .99
                              'authors_relev': .01, # .01
                              # 'words_relev': .2,
                              # 'venues_relev' : .2,
                              'author_affils_relev': .99, # .95, .99, .99, .99
                              'alpha': 0.4}) # .01, .35, .25, .4

            if s.name() == "ProjectedLayered":
                s.set_params(**{
                          'H': 0,
                          'age_relev': .0, # .0
                          'alpha': 0.7, # .7
                          })

            if s.name() == "IterProjectedLayered":
                s.set_params(**{
                          'H': 0,
                          'age_relev': .0, # .0
                          # 'papers_relev': .7, # .99
                          # 'authors_relev': .3, # .01
                          'author_affils_relev': .95, # .95
                          'alpha': .9, # .9, 0.4 (easy_search)
                          'affil_relev': 1.0
                          })

            if s.name() == "StatSearcher":
                s.set_params(**{
                          'H': 0,
                          'age_relev': .0, # .0
                          })

            if s.name() == "TemporalSearcher":
                s.set_params(**{
                          'H': 0,
                          'age_relev': .0, # .0
                          'alpha': .9, # .9, 0.4 (easy_search)
                          })

            rfile = get_results_file(c, "results_%s"%s.name())
            rank_affils(selected_affils, c, year, s, results_file=rfile)
            del s
        bg = Bagging()
        rfile = get_results_file(c, "results_%s"%bg.name())
        rank_affils(selected_affils, c, year, bg,\
                     results_file=rfile, bagging_list=bagging_list)

        print


if __name__ == '__main__':
    main()
    merge_confs_in_phase(2, "Bagging")

