'''
Created on Mar 14, 2016

@author: hugo
'''

import os
import sys
import time
import numpy as np
import config
from collections import defaultdict
from mymysql.mymysql import MyMySQL
from datasets.mag import get_selected_pubs
from ranking.kddcup_searchers import SimpleSearcher, Searcher
from evaluation.kddcup_expt import get_results_file, save_results

db = MyMySQL(db=config.DB_NAME, user=config.DB_USER, passwd=config.DB_PASSWD)



def rank_affils(selected_affils, conf_name, year, searcher, show=True, results_file=None):
    conf_id = db.select("id", "confs", where="abbr_name='%s'"%conf_name, limit=1)[0]
    start = time.time()

    if searcher.name() == "SimpleSearcher":
        results = searcher.search(selected_affils, conf_name, year, age_decay=True, rtype="affil")
        # normalize ranking score
        ids, scores = zip(*results)
        scores = np.array(scores)
        _max = np.max(scores)
        if _max !=  0:
            scores = scores / _max
        results = zip(ids, scores.tolist())
    else:
        results = searcher.search(selected_affils, conf_name, year, force=True, rtype="affil")

    print "Runtime: %s" % (time.time() - start)

    if results_file:
        save_results(conf_id, results, results_file)

def merge_confs_in_phase(phase, method_name):
    confs = [
                [
                    "SIGIR", # Phase 1
                    "SIGMOD",
                    "SIGCOMM"
                ],

                [
                    "KDD", # Phase 2
                    "ICML"
                ],

                [
                    "FSE", # Phase 3
                    "MobiCom",
                    "MM"
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
                "SIGIR", # Phase 1
                "SIGMOD",
                "SIGCOMM",

                # "KDD", # Phase 2
                # "ICML",

                # "FSE", # Phase 3
                # "MobiCom",
                # "MM",
            ]

    searchers = [
                    # SimpleSearcher(),
                    Searcher(**config.PARAMS),

                ]

    # import pdb;pdb.set_trace()
    selected_affils = db.select(fields="id", table="selected_affils")
    year = ["2011", "2012", "2013", "2014", "2015"]

    for c in confs:
        print "Running on '%s' conf." % c

        for s in searchers:
            print "Running %s." % s.name()

            if s.name() == "SimpleSearcher":
                s.set_params(**{
                              'age_relev': .5,
                              })

            if s.name() == "MultiLayered":
                s.set_params(**{
                              'H': 0,
                              # 'age_relev': 0.01, # 0.01
                              'papers_relev': .99, # .99
                              'authors_relev': .01, # .01
                              # 'words_relev': .2,
                              # 'venues_relev' : .2,
                              'author_affils_relev': .95, # .95
                              'alpha': 0.25}) # 0.25

            rfile = get_results_file(c, "results_%s"%s.name())
            rank_affils(selected_affils, c, year, s, results_file=rfile)
            del s
        print


if __name__ == '__main__':
    main()
    merge_confs_in_phase(1, "SimpleSearcher")

