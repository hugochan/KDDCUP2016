'''
Created on Mar 14, 2016

@author: hugo
'''

from datasets.mag import get_selected_pubs
from collections import defaultdict
from ranking.kddcup_ranker import rank_nodes


builder = None

def build_graph(conf_name, year, H, min_topic_lift, min_ngram_lift, exclude=[], force=False, save=True, load=False):
    """
    Utility method to build and return the graph model. First we check if a graph file
    exists. If not, we check if the builder class is already instantiated. If not, we do
    it and proceed to build the graph.
    """
    global builder
    model_folder = config.IN_MODELS_FOLDER % (config.DATASET, H)

    # Creates model folder if non existing
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    graph_file = utils.get_graph_file_name(conf_name, model_folder)
    if force or (not os.path.exists(graph_file)):

        if not builder:
            builder = model.ModelBuilder()

        # Builds the graph file
        graph = builder.build(conf_name, year, H, min_topic_lift, min_ngram_lift, exclude)

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


def simple_search(selected_affils, conf_name, year):
    """
    Parameters
    -----------
    conf_name : A string or list (tuple) of strings
                Specifies targeted conferences

    year : A string or list (tuple) of strings
            Specifies targeted years
    """
    affil_scores = defaultdict()
    pub_records = get_selected_pubs(conf_name, year)

    for _, record in pub_records.iteritems():
        score1 = 1.0 / len(record)
        for _, affil_ids in record.iteritems():
            score2 = score1 / len(affil_ids)
            for each in affil_ids:
                try:
                    affil_scores[each] += score2
                except:
                    affil_scores[each] = score2

    # we only rank the selected affiliations
    if selected_affils:
        selected_affil_scores = defaultdict()
        for each in selected_affils:
            try:
                selected_affil_scores[each] = affil_scores[each]
            except:
                selected_affil_scores[each] = 0.0
    else: # return all the affils
        selected_affil_scores = affil_scores

    selected_affil_scores  = sorted(selected_affil_scores.items(), key=lambda d: d[1], reverse=True)
    return selected_affil_scores


class SimpleSearcher():
    """
    Ranks affliations based on how many of their papers were accepted by
    specific conference in specific period of time.
    """

    def __init__(self):
        pass

    def name(self):
        return "SimpleSearcher"

    def search(self, selected_affils, conf_name, year):
        rst = simple_search(selected_affils, conf_name, year)
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


    def search(self, selected_affils, conf_name, year, exclude=[], rtype="affil", force=False):
        """
        Checks if the graph model already exists, otherwise creates one and
        runs the ranking on the nodes.
        """
        graph = build_graph(conf_name,
                            self.params['H'],
                            year,
                            self.params['min_topic_lift'],
                            self.params['min_ngram_lift'],
                            exclude, force, load=True, save=self.save)

        # Store number of nodes for checking later
        self.nnodes = graph.number_of_nodes()

        # Rank nodes using subgraph
        scores = rank_nodes(graph, selected_affils, return_type=rtype, **self.params)

        # Adds the score to the nodes and writes to disk. A stupid cast
        # is required because write_gexf can't handle np.float64
        scores = {nid: float(score) for nid, score in scores.items()}
        nx.set_node_attributes(graph, "score", scores)

        # nx.write_gexf(graph, utils.get_graph_file_name(model_folder, query))

        # Returns the top values of the type of node of interest
        results = get_top_nodes(graph, scores.items(), limit=limit, return_type=rtype)

        # Add to class object for future access
        self.graph = graph

        return [str(pub_id) for _nid, pub_id, _score in results]

