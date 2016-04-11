'''
Created on Mar 14, 2016

@author: hugo
'''

from datasets.mag import get_selected_expand_pubs
from ranking.kddcup_ranker import rank_nodes, rank_single_layer_nodes, rank_author_affil_nodes, rank_projected_nodes, rank_paper_author_affil_nodes
import kddcup_model
import utils
import config

from collections import defaultdict
import os
import networkx as nx
import numpy as np
from mymysql.mymysql import MyMySQL

from sklearn.linear_model import LinearRegression, BayesianRidge, TheilSenRegressor, RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures



builder = None
db = MyMySQL(db=config.DB_NAME, user=config.DB_USER, passwd=config.DB_PASSWD)


def build_graph(conf_name, year, age_relev, H, alpha, min_topic_lift, min_ngram_lift, alg, exclude=[], force=False, save=True, load=False):
    """
    Utility method to build and return the graph model. First we check if a graph file
    exists. If not, we check if the builder class is already instantiated. If not, we do
    it and proceed to build the graph.
    """
    global builder
    model_folder = config.IN_MODELS_FOLDER % (alg, config.DATASET, H)

    # Creates model folder if non existing
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    graph_file = utils.get_graph_file_name(conf_name, model_folder)
    if force or (not os.path.exists(graph_file)):

        if not builder:
            builder = kddcup_model.ModelBuilder()

        # Builds the graph file
        if alg == 'ProjectedLayered':
            graph = builder.build_affils(conf_name, year, age_relev, H, exclude)

        elif alg == 'MultiLayered':
            graph = builder.build(conf_name, year, H, min_topic_lift, min_ngram_lift, exclude)

        elif alg == 'IterProjectedLayered':
            # graph = builder.build_projected_layers(conf_name, year, age_relev, H, alpha, exclude)
            graph = builder.build_projected_layers2(conf_name, year, age_relev, H, alpha, exclude)

        # Stores gexf copy for caching purposes
        if save:
            nx.write_gexf(graph, graph_file)

        return graph

    else:
        # A gexf copy already exists in disk. Just load it and return
        # print graph_file
        try:
            graph = nx.read_gexf(graph_file, node_type=int)

        except:
            print "Problem opening '%s'." % graph_file
            sys.exit(1)

    return graph


def simple_search(selected_affils, conf_name, year, expand_year=[], age_decay=False, age_relev=0.0):
    """
    Parameters
    -----------
    conf_name : A string or list (tuple) of strings
                Specifies targeted conferences

    year : A string or list (tuple) of strings
            Specifies targeted years
    """
    affil_scores = defaultdict()
    pub_records, _, __ = get_selected_expand_pubs(conf_name, year, _type='selected')

    # expand docs set by getting more papers accepted by the targeted conference
    if expand_year:
        conf_id = db.select("id", "confs", where="abbr_name='%s'"%conf_name, limit=1)[0]
        expand_recrods, _, __ = get_selected_expand_pubs(conf_id, expand_year, _type='expanded')
        pub_records.update(expand_recrods)
        print 'expanded %s papers.'%len(expand_recrods)

    current_year = config.PARAMS['current_year']
    old_year = config.PARAMS['old_year']

    for _, record in pub_records.iteritems():
        if age_decay:
            pub_year = min(max(record['year'], old_year), current_year)
            weight = np.exp(-(age_relev)*(current_year-pub_year))
        else:
            weight = 1.0

        score1 = weight / len(record['author'])
        for author_id, affil_ids in record['author'].iteritems():
            if affil_ids:
                score2 = score1 / len(affil_ids)
                for each in affil_ids:
                    try:
                        affil_scores[each] += score2
                    except:
                        affil_scores[each] = score2

    # we only rank the selected affiliations
    if selected_affils:
        selected_affil_scores = get_selected_nodes(affil_scores, selected_affils)

    else: # sort and return all the affils
        selected_affil_scores = sorted(affil_scores.items(), key=lambda d: d[1], reverse=True)

    return selected_affil_scores



def regression_search(selected_affils, conf_name, year, expand_year=[]):
    """
    Parameters
    -----------
    conf_name : A string or list (tuple) of strings
                Specifies targeted conferences

    year : A string or list (tuple) of strings
            Specifies targeted years
    """
    affil_scores = defaultdict()
    pub_records, _, __ = get_selected_expand_pubs(conf_name, year, _type='selected')

    # expand docs set by getting more papers accepted by the targeted conference
    if expand_year:
        conf_id = db.select("id", "confs", where="abbr_name='%s'"%conf_name, limit=1)[0]
        expand_recrods, _, __ = get_selected_expand_pubs(conf_id, expand_year, _type='expanded')
        pub_records.update(expand_recrods)
        print 'expanded %s papers.'%len(expand_recrods)

    current_year = config.PARAMS['current_year']
    old_year = config.PARAMS['old_year']


    for _, record in pub_records.iteritems():
        pub_year = min(max(record['year'], old_year), current_year)

        score1 = 1.0 / len(record['author'])
        for author_id, affil_ids in record['author'].iteritems():
            if affil_ids:
                score2 = score1 / len(affil_ids)
                for each in affil_ids:
                    if affil_scores.has_key(each):
                        try:
                            affil_scores[each][pub_year] += score2
                        except:
                            affil_scores[each][pub_year] = score2
                    else:
                        affil_scores[each] = {pub_year: score2}

    # regression
    years_set = sorted(expand_year + year)
    affils = affil_scores.keys()
    training_data = []
    target = []
    for each_affil in affils:
        sample = []
        for each_year in years_set[:-1]:
            try:
                sample.append(affil_scores[each_affil][int(each_year)])
            except:
                sample.append(0)
        try:
            target.append(affil_scores[each_affil][int(years_set[-1])])
        except:
            target.append(.0)
        training_data.append(sample)

    training_data = np.array(training_data)
    target = np.array(target)

    poly = PolynomialFeatures(3)
    new_features = poly.fit_transform(training_data)

    lr = LinearRegression()
    # lr = BayesianRidge()
    # lr = TheilSenRegressor(random_state=42)
    # lr = RANSACRegressor(random_state=42)
    lr.fit(new_features, target)

    test_data = np.hstack((training_data[:, 1:], target.reshape(target.shape[0], 1)))
    test_data = poly.fit_transform(test_data)

    preds = lr.predict(test_data)
    pred_affil_scores = dict(zip(affils, preds))
    print "score: %s" % lr.score(new_features, target)

    # we only rank the selected affiliations
    if selected_affils:
        selected_affil_scores = get_selected_nodes(pred_affil_scores, selected_affils)

    else: # sort and return all the affils
        selected_affil_scores = sorted(pred_affil_scores.items(), key=lambda d: d[1], reverse=True)

    return selected_affil_scores



class SimpleSearcher():
    """
    Ranks affliations based on how many of their papers were accepted by
    specific conference in specific period of time.
    """

    def __init__(self, **params):
        self.params = params

    def name(self):
        return "SimpleSearcher"

    def set_param(self, name, value):
        self.params[name] = value

    def set_params(self, **params):
        for k, v in params.items():
            self.params[k] = v

    def search(self, selected_affils, conf_name, year, expand_year=[], age_decay=False, rtype="affil"):
        rst = simple_search(selected_affils, conf_name, year, expand_year=expand_year, age_decay=age_decay, age_relev=self.params['age_relev'])
        return rst


class RegressionSearcher():
    """
    Ranks affliations based on how many of their papers were accepted by
    specific conference in specific period of time.
    """

    def __init__(self, **params):
        self.params = params

    def name(self):
        return "RegressionSearcher"

    def set_param(self, name, value):
        self.params[name] = value

    def set_params(self, **params):
        for k, v in params.items():
            self.params[k] = v

    def search(self, selected_affils, conf_name, year, expand_year=[], age_decay=False, rtype="affil"):
        rst = regression_search(selected_affils, conf_name, year, expand_year=expand_year)
        return rst


class Searcher:
    """
    Basic searcher class for the Multi-Layered method.
    """

    def __init__(self, **params):
        self.params = params
        self.save = True

    def name(self):
        return "MultiLayered"

    def set_save(self, save):
        self.save = save

    def set_params(self, **params):
        for k, v in params.items():
            self.params[k] = v

    def set_param(self, name, value):
        self.params[name] = value

    def number_of_nodes(self):
        """ Return number of nodes if search was called once. """
        return self.nnodes

    def get_nx_graph(self):
        """ Return networkx graph structured if search method was called once. """
        return self.graph


    def search(self, selected_affils, conf_name, year, exclude_papers=[], rtype="affil", force=False):
        """
        Checks if the graph model already exists, otherwise creates one and
        runs the ranking on the nodes.
        """
        graph = build_graph(conf_name,
                            year,
                            None,
                            self.params['H'],
                            self.params['min_topic_lift'],
                            self.params['min_ngram_lift'],
                            self.name(), exclude_papers, force, load=True, save=self.save)

        # Store number of nodes for checking later
        self.nnodes = graph.number_of_nodes()

        # Rank nodes using subgraph
        scores = rank_nodes(graph, return_type=rtype, **self.params)

        # Adds the score to the nodes and writes to disk. A stupid cast
        # is required because write_gexf can't handle np.float64
        scores = {nid: float(score) for nid, score in scores.items()}
        nx.set_node_attributes(graph, "score", scores)

        # nx.write_gexf(graph, utils.get_graph_file_name(model_folder, query))

        # Returns the top values of the type of node of interest
        results = get_top_nodes(graph, scores.items(), limit=selected_affils, return_type=rtype)

        # Add to class object for future access
        self.graph = graph

        return results


class ProjectedSearcher:
    """
    Basic searcher class for the Projected-Layer method.
    """

    def __init__(self, **params):
        self.params = params
        self.save = True

    def name(self):
        return "ProjectedLayered"

    def set_save(self, save):
        self.save = save

    def set_params(self, **params):
        for k, v in params.items():
            self.params[k] = v

    def set_param(self, name, value):
        self.params[name] = value

    def number_of_nodes(self):
        """ Return number of nodes if search was called once. """
        return self.nnodes

    def get_nx_graph(self):
        """ Return networkx graph structured if search method was called once. """
        return self.graph


    def search(self, selected_affils, conf_name, year, exclude_papers=[], rtype="affil", force=False):
        """
        Checks if the graph model already exists, otherwise creates one and
        runs the ranking on the nodes.
        """
        graph = build_graph(conf_name,
                            year,
                            self.params['age_relev'],
                            self.params['H'],
                            self.params['min_topic_lift'],
                            self.params['min_ngram_lift'],
                            self.name(), exclude_papers, force, load=True, save=self.save)

        # Store number of nodes for checking later
        self.nnodes = graph.number_of_nodes()

        # Rank nodes using subgraph
        scores = rank_single_layer_nodes(graph, **self.params)

        # Adds the score to the nodes and writes to disk. A stupid cast
        # is required because write_gexf can't handle np.float64
        scores = {nid: float(score) for nid, score in scores.items()}
        nx.set_node_attributes(graph, "score", scores)

        # nx.write_gexf(graph, utils.get_graph_file_name(model_folder, query))

        # Returns the top values of the type of node of interest
        results = get_top_nodes(graph, scores.items(), limit=selected_affils, return_type=rtype)

        # Add to class object for future access
        self.graph = graph

        return results


class IterProjectedSearcher:
    """
    Basic searcher class for the Iterative-Projected-Layer method.
    Two-stage projection: Paper -> Author -> Affil
    """

    def __init__(self, **params):
        self.params = params
        self.save = True

    def name(self):
        return "IterProjectedLayered"

    def set_save(self, save):
        self.save = save

    def set_params(self, **params):
        for k, v in params.items():
            self.params[k] = v

    def set_param(self, name, value):
        self.params[name] = value

    def number_of_nodes(self):
        """ Return number of nodes if search was called once. """
        return self.nnodes

    def get_nx_graph(self):
        """ Return networkx graph structured if search method was called once. """
        return self.graph



    def easy_search(self, selected_affils, conf_name, year, exclude_papers=[]):
        """

        """
        builder = kddcup_model.ModelBuilder()

        # scores = builder.get_ranked_affils_by_papers(conf_name, year, self.params['age_relev'], self.params['H'], self.params['alpha'], exclude=exclude_papers)
        scores = builder.get_ranked_affils_by_authors(conf_name, year, self.params['age_relev'], self.params['H'], self.params['alpha'], exclude=exclude_papers)

        results = get_selected_nodes(scores, selected_affils)

        return results


    def search(self, selected_affils, conf_name, year, exclude_papers=[], rtype="affil", force=False):
        """
        Checks if the graph model already exists, otherwise creates one and
        runs the ranking on the nodes.
        """
        graph = build_graph(conf_name,
                            year,
                            self.params['age_relev'],
                            self.params['H'],
                            self.params['alpha'],
                            self.params['min_topic_lift'],
                            self.params['min_ngram_lift'],
                            self.name(), exclude_papers, force, load=True, save=self.save)

        # Store number of nodes for checking later
        self.nnodes = graph.number_of_nodes()

        # Rank nodes using subgraph
        # scores = rank_single_layer_nodes(graph, **self.params)
        scores = rank_author_affil_nodes(graph, return_type=rtype, **self.params)
        # scores = rank_paper_author_affil_nodes(graph, return_type=rtype, **self.params)

        # Adds the score to the nodes and writes to disk. A stupid cast
        # is required because write_gexf can't handle np.float64
        scores = {nid: float(score) for nid, score in scores.items()}
        nx.set_node_attributes(graph, "score", scores)

        # nx.write_gexf(graph, utils.get_graph_file_name(model_folder, query))

        # Returns the top values of the type of node of interest
        results = get_top_nodes(graph, scores.items(), limit=selected_affils, return_type=rtype)

        # Add to class object for future access
        self.graph = graph

        return results





def get_top_nodes(graph, scores, limit, return_type="affil"):
    """
    Helper method that takes the graph and the calculated scores and outputs a sorted rank
    of the request type of node.
    """

    related_scores = dict([(graph.node[k]["entity_id"], v) for k, v in scores\
                     if graph.node[k]["type"] == return_type])


    ranking = get_selected_nodes(related_scores, limit)

    return ranking


def get_selected_nodes(ranking, selected_nodes):
    """
    Helper method that ranks the selected nodes given the ranking over all the nodes.
    """

    selected_ranking = defaultdict()
    for each in selected_nodes:
        try:
            selected_ranking[each] = ranking[each]
        except:
            selected_ranking[each] = 0.0

    selected_ranking  = sorted(selected_ranking.items(), key=lambda d: d[1], reverse=True)
    return selected_ranking
