'''
Created on May 19, 2015

@author: luamct
'''

from ranking import model
import logging as log
from ranking.model import get_all_edges
import random
import networkx as nx
import config
import os
from ranking import ranker
import utils
from ranking.searchers import get_top_nodes


class MengModelBuilder(model.ModelBuilder) :

  def get_pubs_layer(self, sample=None) :
    edges = []
    nodes = set()
    for u, v in get_all_edges() :
      u = str(u)
      v = str(v)
      nodes.add(u)
      nodes.add(v)
      edges.append((u, v, 1.0))


    if sample :
      snodes = set(random.sample(list(nodes), sample))
      sedges = []
      for u, v, w in edges :
        if (u in snodes) and (v in snodes) :
          sedges.append((u,v,w))

      nodes = snodes
      edges = sedges

    return list(nodes), edges


  def build(self) :
    '''
    Build the full graph according to the paper description.
    '''
    # Assemble model file name if not provided
#		if not model_file :
#			model_file = "models/%s_N%d_H%d.gexf" % (query, n_starting_nodes, n_hops)

    log.info("Building full graph model.")
#		log.info("Writing model into file '%s'" % model_file)
#		self.make_cache_folder(query, n_starting_nodes, n_hops)

    pubs, citation_edges = self.get_pubs_layer()
    log.info("%d pubs and %d citation edges." % (len(pubs), len(citation_edges)))

    authors, coauth_edges, auth_edges = self.get_authors_layer(pubs)

    # Change coauthorship edge weights to 1
#		_coauth_edges_ = [(a1, a2, 1.0) for a1, a2, _w in coauth_edges]

    log.info("%d authors, %d co-authorship edges and %d authorship edges." % (len(authors), len(coauth_edges), len(auth_edges)))

    topics, topic_topic_edges, pub_topic_edges = self.get_topics_layer_from_db(pubs, 1.0)
    log.info("%d topics, %d topic-topic edges (not used) and %d pub-topic edges."
                    % (len(topics), len(topic_topic_edges), len(pub_topic_edges)))

#		word_nodes, paper_word_edges = self.get_words_layer_from_db(papers)
    words, _word_word_edges, pub_word_edges = self.get_ngrams_layer_from_db(pubs, 1.0)
    log.info("%d words and %d pub-word edges." % (len(words), len(pub_word_edges)))

    graph = self.assemble_layers(pubs, citation_edges, authors, coauth_edges, auth_edges,
                            topics, pub_topic_edges, words, pub_word_edges)

    log.info("Total of %d nodes and %d edges" % (graph.number_of_nodes(), graph.number_of_edges()))
    return graph


  def assemble_layers(self, pubs, citation_edges, authors, coauth_edges, auth_edges,
                      topics, paper_topic_edges, ngrams, paper_ngram_edges) :
    '''
    Assembles the layers as an unified graph. Each node as an unique id, its type (paper,
    author, etc.) and a readable label (paper title, author name, etc.)
    '''
    graph = nx.DiGraph()

    # These map the original identifiers for each type (paper doi, author id,
    # etc.) to the new unique nodes id.
    pubs_ids = {}
    authors_ids = {}
    topics_ids = {}
    words_ids = {}

    # Controls the unique incremental id generation
    next_id = 0

    # Add each paper providing an unique node id. Some attributes must be added
    # even if include_attributes is True, since they are used in ranking algorithm.
    for pub in pubs :
      pub = str(pub)

      # Add node to graph including some necessary attributes
      graph.add_node(next_id,
                     type="paper",
                     entity_id=pub,
                     year=self.pub_years[pub])

      pubs_ids[pub] = next_id
      next_id += 1

    # Add citation edges (directed)
    for paper1, paper2, _weight in citation_edges :
      graph.add_edge(pubs_ids[paper1], pubs_ids[paper2], weight=1.0)


    # Add each author providing an unique node id
    for author in authors :
      graph.add_node(next_id, type="author", entity_id=author)

      authors_ids[author] = next_id
      next_id += 1


    # Add co-authorship edges on both directions (undirected)
    for author1, author2, _weight in coauth_edges :
      graph.add_edge(authors_ids[author1], authors_ids[author2], weight=1.0)
      graph.add_edge(authors_ids[author2], authors_ids[author1], weight=1.0)

    # Add authorship edges on both directions (undirected)
    for paper, author, _weight in auth_edges :
      graph.add_edge(pubs_ids[paper], authors_ids[author], weight=1.0)
      graph.add_edge(authors_ids[author], pubs_ids[paper], weight=1.0)


    # Add topic nodes
    for topic in topics :
      graph.add_node(next_id, type="topic", entity_id=topic)

      topics_ids[topic] = next_id
      next_id += 1

    # Add paper-topic edges (directed)
    for paper, topic, _weight in paper_topic_edges :
      graph.add_edge(pubs_ids[paper], topics_ids[topic], weight=1.0)
      graph.add_edge(topics_ids[topic], pubs_ids[paper], weight=1.0)

    ####################################
    # Add ngram nodes
    for ngram in ngrams :
      graph.add_node(next_id, type="ngram", entity_id=ngram)

      words_ids[ngram] = next_id
      next_id += 1

    # Add paper-word edges (undirected)
    for paper, word, _weight in paper_ngram_edges :
      graph.add_edge(pubs_ids[paper], words_ids[word], weight=1.0)
      graph.add_edge(words_ids[word], pubs_ids[paper], weight=1.0)

    return graph



class MengSearcher :
  '''
  Basic searcher class for the Multi-Layered method.
  '''

  def __init__(self, **params) :
    self.params = params

    if not os.path.exists(config.MENG_GRAPH_PATH) :
      log.debug("Meng graph file not found. Building one at '%s'" % config.MENG_GRAPH_PATH)

      mb = MengModelBuilder()
      self.graph = mb.build()
      del mb

      log.debug("Meng graph built. %d nodes and %d edges."
               % (self.graph.number_of_nodes(), self.graph.number_of_edges()))

      utils.ensure_folder(os.path.dirname(config.MENG_GRAPH_PATH))
      nx.write_gexf(self.graph, config.MENG_GRAPH_PATH)

      log.debug("Meng graph saved.")

    else:

      log.debug("Reading Meng graph file at '%s'" % config.MENG_GRAPH_PATH)
      self.graph = nx.read_gexf(config.MENG_GRAPH_PATH, node_type=int)


  def name(self):
    return "Meng"

  def set_params(self, **params) :
    for k, v in params.items() :
      self.params[k] = v

  def set_param(self, name, value):
    self.params[name] = value


  def number_of_nodes(self) :
    ''' Return number of nodes if search was called at least once. '''
    return self.nnodes

  def get_nx_graph(self):
    ''' Return networkx graph structured if search method was called once. '''
    return self.graph


  def search(self, query, exclude=[], limit=50, rtype="paper", force=False):
    '''
    Checks if the graph model already exists, otherwise creates one and
    runs the ranking on the nodes.
    '''

    query_tokens = set(utils.tokenize(query))

    pers = {}
    for nid, atts in self.graph.nodes(data=True) :
      if atts['type']=='ngram' :
        if atts['entity_id'] in query_tokens :
          pers[nid] = 1.0
          continue

      pers[nid] = 0.0


    # Rank nodes using subgraph
    scores, _ = ranker.pagerank(self.graph, alpha=0.85, pers=pers, nstart=pers)

    # Returns the top values of the type of node of interest
    results = get_top_nodes(self.graph, scores.items(), limit=limit, return_type="paper")

    return [str(pub_id) for _nid, pub_id, _score in results]



if __name__ == '__main__':
  log.basicConfig(format='%(asctime)s [%(levelname)s] : %(message)s', level=log.INFO)

  s = MengSearcher()
  r = s.search("implicit learning")
  print "\n".join(r)

#	mb = MengModelBuilder()
#	mb.build()

