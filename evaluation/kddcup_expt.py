'''
Created on Mar 13, 2016

@author: hugo
'''
from mymysql.mymysql import MyMySQL
import config
from collections import defaultdict
import time

# import logging as log
import os
import cPickle
# from baselines.scholar import match_by_title
from evaluation.metrics import ndcg2
from datasets.mag import get_selected_docs
from ranking.kddcup_searchers import simple_search, SimpleSearcher, Searcher


# log.basicConfig(format='%(asctime)s [%(levelname)s] : %(message)s', level=log.INFO)

db = MyMySQL(db=config.DB_NAME, user=config.DB_USER, passwd=config.DB_PASSWD)


def calc_ground_truth_score(selected_affils, conf_name, year="2015"): # {paper_id: {author_id:[affil_id,],},}
    """
    Uses latest pub records to estimate ground truth scores.
    """
    ground_truth = simple_search(selected_affils, conf_name, year)

    return ground_truth

def calc_ndcg(ground_truth, pred):
    actual, relevs = zip(*ground_truth)
    metric = ndcg2(actual, pred, relevs, k=len(actual))

    return metric

def get_results_file(conf_name, method_name) :
    folder = "%sresults/%s" % (config.DATA, conf_name)
    if not os.path.exists(folder) :
        os.makedirs(folder)

    return "%s/%s.tsv" % (folder, method_name)


def save_results(conf_id, results, file_path) :
    '''
    Saves the results in a output file.
    '''
    # cPickle.dump(results, open(file_path, 'w'))
    # write to tsv file
    try:
        f = open(file_path, 'w+')
    except Exception, e:
        print e
        return

    for affil_id, score in results:
        # [conference id] \t [affiliation id] \t [probability score] \n
        f.writelines("%s\t%s\t%s\n"%(conf_id, affil_id, score))
    f.close()


def get_search_metrics(selected_affils, ground_truth, conf_name, year, searcher, exclude_papers=[], show=True, results_file=None) :
    '''
    Run searches on each conference (conference -> ground truth) and return
    the evaluate metric for each instance. Right now the metrics being
    returned is NDCG.

    Returns: dict {metric: value}
    '''
    metrics = defaultdict(list)
    conf_id = db.select("id", "confs", where="abbr_name='%s'"%conf_name, limit=1)[0]
    start = time.time()

    if searcher.name() == "SimpleSearcher":
        results = searcher.search(selected_affils, conf_name, year, rtype="affil")
    else:
        results = searcher.search(selected_affils, conf_name, year, exclude_papers, rtype="affil")

    metrics["Time"] = time.time() - start

    actual, relevs = zip(*ground_truth)
    pred = zip(*results)[0]

    metrics["NDCG"] = ndcg2(actual, pred, relevs, k=len(actual))


    if results_file:
        save_results(conf_id, results, results_file)

    if show:
        for k, v in metrics.iteritems():
            print u"%s: %.1f\t" % (k, v)
        print

    return metrics

def main():

    confs = [
                "SIGIR", # Phase 1
                # "SIGMOD",
                # "SIGCOMM",

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
    year = ["2011", "2012", "2013", "2014"]
    for c in confs :
        # log.info("Running '%s' conf.\n" % c)
        print "Running on '%s' conf." % c
        ground_truth = calc_ground_truth_score(selected_affils, c)
        exclude_papers = get_selected_docs(c, "2015")

        for s in searchers :
            print "Running %s." % s.name()

            if s.name() == "MultiLayered":
                s.set_params(**{
                              'H': 1,
                              # 'age_relev': 0.01, # 0.01
                              'papers_relev': .25,
                              'authors_relev': .25,
                              'words_relev': .25,
                              # 'venues_relev' : .2,
                              'affils_relev': .25,
                              'alpha': 0.3}) # 0.1

            rfile = get_results_file(c, s.name())
            get_search_metrics(selected_affils, ground_truth, c, year, s,\
                         exclude_papers=exclude_papers, results_file=rfile)
            del s
        print


if __name__ == "__main__":
    main()
