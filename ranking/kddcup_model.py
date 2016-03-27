"""
Created on Mar 14, 2016

@author: hugo
"""

import chardet
import numpy as np
import networkx as nx
from mymysql import MyMySQL
from collections import defaultdict
from exceptions import TypeError

# from pylucene import Index
import itertools
import os
import logging as log
import words
import config
import utils
from datasets.mag import get_selected_docs, get_conf_docs, retrieve_affils_by_authors



# Database connection
db = MyMySQL(db=config.DB_NAME,
             user=config.DB_USER,
             passwd=config.DB_PASSWD)


def get_all_edges(papers):
  """
  Retrieve all edges related to given papers from the database.
  """
  if hasattr(papers, '__iter__'):
    if len(papers) == 0:
      return []
    else:
      paper_str = ",".join(["'%s'" % paper_id for paper_id in papers])
  else:
      raise TypeError("Parameter 'papers' is of unsupported type. Iterable needed.")

  rows = db.select(fields=["paper_id", "paper_ref_id"], table="paper_refs",
            where="paper_id IN (%s) OR paper_ref_id IN (%s)"%(paper_str, paper_str))

  return rows


def show_stats(graph):
  print "%d nodes and %d edges." % (graph.number_of_nodes(), graph.number_of_edges())


def write_graph(graph, outfile):
  """
  Write the networkx graph into a file in the gexf format.
  """
  log.info("Dumping graph: %d nodes and %d edges." % (graph.number_of_nodes(), graph.number_of_edges()))
  nx.write_gexf(graph, outfile, encoding="utf-8")


def get_paper_year(paper_id):
  """
  Returns the year of the given paper as stored in the DB.
  """
  year = db.select_one(fields="year", table="papers", where="id='%s'" % paper_id)
  return (int(year) if year else 0)


# def show(doc_id):
#   """ Small utility method to show a document on the browser. """
#   from subprocess import call, PIPE

#   call(["google-chrome", "--incognito", "/data/pdf/%s.pdf" % doc_id], stdout=PIPE, stderr=PIPE)


def add_attributes(graph, entities, node_ids, atts):
  """
  Adds attributes to the nodes associated to the given entities (papers, authors, etc.)
  """
  for entity in entities:
    graph.add_node(node_ids[entity], **atts[entity])


def normalize_edges(edges):
  """
  Normalize the weight on given edges dividing by the maximum weight found.
  """
  wmax = 0.0
  for _u, _v, w in edges:
    wmax = max(w, wmax)

  return [(u, v, w / float(wmax)) for u, v, w in edges]


# def similarity(d1, d2):
#   """
#   Cosine similarity between sparse vectors represented as dictionaries.
#   """
#   sim = 0.0
#   for k in d1:
#     if k in d2:
#       sim += d1[k] * d2[k]

#   dem = np.sqrt(np.square(d1.values()).sum()) * np.sqrt(np.square(d2.values()).sum())
#   return sim / dem


def sorted_tuple(a, b):
  """ Simple pair sorting to avoid repetitions when inserting into set or dict. """
  return (a, b) if a < b else (b, a)


def get_rules_by_lift(transactions, min_lift=1.0):
  """
  Get strong rules from transactions and minimum lift provided.
  """
  freqs1 = defaultdict(int)  # Frequencies of 1-itemsets
  freqs2 = defaultdict(int)  # Frequencies of 2-itemsets
  for trans in transactions:
    for i in trans:
      freqs1[i] += 1

    # If there are at least 2 items, let's compute pairs support
    if len(trans) >= 2:
      for i1, i2 in itertools.combinations(trans, 2):
        freqs2[sorted_tuple(i1, i2)] += 1

  n = float(len(transactions))

  # Check every co-occurring ngram
  rules = []
  for (i1, i2), f in freqs2.items():

    # Consider only the ones that appear more than once together,
    # otherwise lift values can be huge and not really significant
    if f >= 1:
      lift = f * n / (freqs1[i1] * freqs1[i2])

      # Include only values higher than min_lift
      if lift >= min_lift:
        rules.append((i1, i2, lift))

  return rules


########################################
## Class definitions
########################################


class GraphBuilder:
  """
  Graph structure designed to store edges and operate efficiently on some specific
  graph building and expanding operations.
  """

  def __init__(self, edges):

    self.citing = defaultdict(list)
    self.cited = defaultdict(list)

    for f, t in edges:
      f = str(f).strip('\r\n')
      t = str(t).strip('\r\n')
      self.citing[f].append(t)
      self.cited[t].append(f)


  def follow_nodes(self, nodes):
    """
    Return all nodes one edge away from the given nodes.
    """
    new_nodes = set()
    for n in nodes:
      new_nodes.update(self.citing[n])
      new_nodes.update(self.cited[n])

    return new_nodes


  def subgraph(self, nodes):
    """
    Return all edges between the given nodes.
    """
    # Make sure lookup is efficient
    nodes = set(nodes)

    new_edges = []
    for n in nodes:

      for cited in self.citing[n]:
        if (n != cited) and (cited in nodes):
          new_edges.append((n, cited))

      for citing in self.cited[n]:
        if (n != citing) and (citing in nodes):
          new_edges.append((citing, n))

    return set(new_edges)


class ModelBuilder:
  """
  Main class for building the graphical model. The layers are built separately in their
  corresponding methods. Every layer is cached in a folder defined by the main parameters.
  """

  def __init__(self, include_attributes=False):
    """
    Initializes structures and load data into memory, such as the text index and
    the citation graph.
    """
    # # Build text index if non-existing
    # if not os.path.exists(config.INDEX_PATH):
    #   indexer = Indexer()
    #   indexer.add_papers(config.INDEX_PATH, include_text=False)

    # # Load text index
    # self.index = Index(config.INDEX_PATH, similarity="tfidf")

    # Graph structure that allows fast access to nodes and edges
    # self.edges_lookup = GraphBuilder(get_all_edges())

    # If attributes should be fetched and included in the model for each type of node.
    # Should be true for visualization and false for pure relevance calculation.
    self.include_attributes = include_attributes
    self.pub_years = defaultdict()

    # Create a helper boolean to check if citation contexts are
    # going to be used (some datasets don't have it available)
    # self.use_contexts = (config.DATASET == 'csx')

    # Load vocabulary for the tokens in the citation contexts
    # if self.use_contexts:
    #   self.ctxs_vocab, self.nctx = words.read_vocab(config.CTXS_VOCAB_PATH)

    log.debug("ModelBuilder constructed.")


  def get_weights_file(self, edges):
    return [(u, v, 1.0) for (u, v) in edges]


  def get_pubs_layer(self, conf_name, year, n_hops, exclude_list=[], expand_method='n_hops'):
    """
    First documents are retrieved from pub records of a targeted conference.
    Then we follow n_hops from these nodes to have the first layer of the graph (papers).
    """

    # Fetches all document that have at least one of the terms
    pubs = get_selected_docs(conf_name, year)
    docs = zip(*pubs)[0]
    # add year
    self.pub_years = dict(pubs)

    if expand_method == 'n_hops':

      # Get doc ids as uni-dimensional list
      self.edges_lookup = GraphBuilder(get_all_edges(docs))
      nodes = set(docs)

      # Expand the docs set by reference
      nodes = self.get_expanded_pubs_by_nhops(nodes, self.edges_lookup, exclude_list, n_hops)


    elif expand_method == 'conf':

      # Expand the docs by getting more papers from the targeted conference
      # expanded_pubs = self.get_expanded_pubs_by_conf(conf_name, [2009, 2010])
      nodes = set(docs)
      expanded_pubs = self.get_expanded_pubs_by_conf2(conf_name, range(2006, 2011))

      # add year
      for paper, year in expanded_pubs:
        self.pub_years[paper] = year

      # Remove documents from the exclude list and keep only processed ids
      expanded_docs = set(zip(*expanded_pubs)[0]) - set(exclude_list)
      nodes.update(expanded_docs)

      self.edges_lookup = GraphBuilder(get_all_edges(nodes))

    else:
      raise ValueError("parameter expand_method should either be n_hops or conf.")


    # Get the edges between the given nodes and add a constant the weight for each
    edges = self.edges_lookup.subgraph(nodes)

    # Get edge weights according to textual similarity between
    # the query and the citation context
    weighted_edges = self.get_weights_file(edges)

    # To list to preserve element order
    nodes = list(nodes)

    # Save into cache for reusing
    #       cPickle.dump((nodes, edges, self.query_sims), open(cache_file, 'w'))

    return nodes, weighted_edges


  def get_expanded_pubs_by_conf(self, conf_name, year):
    # Expand the docs by getting more papers from the targeted conference
    conf_id = db.select("id", "confs", where="abbr_name='%s'"%conf_name, limit=1)[0]
    expanded_pubs = get_conf_docs(conf_id, year)

    return expanded_pubs

  def get_expanded_pubs_by_conf2(self, conf_name, year):
    # Expand the docs by getting more papers from the targeted conference
    conf_id = db.select("id", "confs", where="abbr_name='%s'"%conf_name, limit=1)[0]

    year_str = ",".join(["'%s'" % y for y in year])
    expanded_pubs = db.select(["paper_id", "year"], "expanded_conf_papers", where="conf_id='%s' and year IN (%s)"%(conf_id, year_str))

    return expanded_pubs


  def get_expanded_pubs_by_nhops(self, nodes, edges_lookup, exclude_list, n_hops):
    new_nodes = nodes

    # We hop h times including all the nodes from these hops
    for h in xrange(n_hops):
      new_nodes = edges_lookup.follow_nodes(new_nodes)

      # Remove documents from the exclude list and keep only processed ids
      new_nodes -= set(exclude_list)

      # Than add them to the current set
      nodes.update(new_nodes)

      log.debug("Hop %d: %d nodes." % (h + 1, len(nodes)))

    return nodes


  def get_paper_based_coauthorships(self, papers, weighted=True):
    """
    Return all the collaborationships between the given papers. Edges to authors not
    provided (according to given papers) are not included.
    """

    if hasattr(papers, '__iter__'):
      if len(papers) == 0:
        return []
      else:
        paper_str = ",".join(["'%s'" % paper_id for paper_id in papers])
    else:
      raise TypeError("Parameter 'papers' is of unsupported type. Iterable needed.")


    rows = db.select(["paper_id", "author_id"], "paper_author_affils", where="paper_id IN (%s)"%paper_str)

    rows = set(rows) # Removing duplicate records
    author_papers = defaultdict()
    for paper, author in rows:
      try:
        author_papers[author].add(paper)
      except:
        author_papers[author] = set([paper])

    coauthorships = []
    authors = author_papers.keys()
    for i in xrange(len(authors) - 1):
      for j in xrange(i + 1, len(authors)):
        npapers = float(len(author_papers[authors[i]] & author_papers[authors[j]]))

        if npapers > 0:
          # Apply log transformation to smooth values and avoid outliers
          npapers = 1.0 + np.log(npapers) if weighted else 1.0

          coauthorships.append((authors[i], authors[j], npapers))

    return authors, rows, coauthorships


  def get_authors(self, doc_id):
    """
    Return the authors associated with the given paper, if available.
    """
    return db.select("author_id", table="paper_author_affils", where="paper_id='%s'" % doc_id)


  def get_cached_coauthorship_edges(self, authors):
    """
    Return all the collaboration edges between the given authors. Edges to authors not provided are
    not included.
    """
    # For efficient lookup
    authors = set(authors)

    edges = set()
    for author_id in authors:
      coauthors = db.select(["author1", "author2", "npapers"], "coauthorships",
                  where="author1=%d OR author2=%d" % (author_id, author_id))
      for a1, a2, npapers in coauthors:

        # Apply log transformation to smooth values and avoid outliers
        # crushing other values after normalization
        npapers = 1.0 + np.log(npapers)

        if (a1 in authors) and (a2 in authors):
          edge = (a1, a2, 1.0) if a1 < a2 else (a2, a1, 1.0)
          edges.add(edge)

    # Normalize by max value and return them as a list
    return normalize_edges(edges)


  def get_coauthorship_edges(self, authors):
    """
    Return all the collaboration edges between the given authors. Edges to authors not provided are
    not included.
    """
    # For efficient lookup
    authors = set(authors)

    edges = set()
    for author_id in authors:
      coauthorships = db.select_query("""SELECT b.author_id FROM authorships a, authorships b
                                         WHERE (a.author_id=%d) AND (b.author_id!=%d) AND a.paper_id=b.paper_id""" \
                      % (author_id, author_id))

      # Count coauthorshiped pubs
      coauthors = defaultdict(int)
      for (coauthor,) in coauthorships:
        if coauthor in authors:
          coauthors[(author_id, coauthor)] += 1

      for (a1, a2), npapers in coauthors.items():

        # Apply log transformation to smooth values and avoid outliers
        # crushing other values after normalization
        weight = 1.0 + np.log(npapers)

        if (a1 in authors) and (a2 in authors):
          edge = (a1, a2, weight) if a1 < a2 else (a2, a1, weight)
          edges.add(edge)

    # Normalize by max value and return them as a list
    return normalize_edges(edges)


  def get_authorship_edges(self, papers_authors):
    """
    Return authorship edges [(doc_id, author), ...]
    """
    edges = []
    for doc_id, authors in papers_authors.items():
      edges.extend([(doc_id, author, 1.0) for author in authors])

    return edges


  def get_authors_layer(self, papers, ign_cache=False):
    """
    Retrieve relevant authors from DB (author of at least one paper given as argument)
    and assemble co-authorship and authorship nodes and edges.
    """

    # Try to load from cache
    #       cache_file = "%s/authors.p" % self.cache_folder
    #       if (not ign_cache) and os.path.exists(cache_file) :
    #           return cPickle.load(open(cache_file, 'r'))

    authors, auth_edges, coauth_edges = self.get_paper_based_coauthorships(papers)

    # Save into cache for reuse
    #       cPickle.dump((all_authors, coauth_edges, auth_edges), open(cache_file, 'w'))

    return authors, coauth_edges, auth_edges


  def get_relevant_topics(self, doc_topics, ntop=None, above=None):
    """
    Get the most important topics for the given document by either:
      * Taking the 'ntop' values if 'ntop' id provided or
      * Taking all topics with contributions greater than 'above'.
    """
    if ntop:
      return np.argsort(doc_topics)[::-1][:ntop]

    if above:
      return np.where(doc_topics > above)[0]

    raise TypeError("Arguments 'ntop' and 'above' cannot be both None.")


  def get_frequent_topic_pairs(self, topics_per_document, min_interest):

    freqs1 = defaultdict(int)  # Frequencies of 1-itemsets
    freqs2 = defaultdict(int)  # Frequencies of 2-itemsets
    for topics in topics_per_document:
      for t in topics:
        freqs1[t] += 1

      if len(topics) >= 2:
        for t1, t2 in itertools.combinations(topics, 2):
          freqs2[sorted_tuple(t1, t2)] += 1

    total = float(len(topics_per_document))

    rules = []
    for (t1, t2), v in sorted(freqs2.items(), reverse=True, key=lambda (k, v): v):

      int12 = float(v) / freqs1[t1] - freqs1[t2] / total
      int21 = float(v) / freqs1[t2] - freqs1[t1] / total

      if int12 >= min_interest: rules.append((t1, t2, int12))
      if int21 >= min_interest: rules.append((t2, t1, int21))

    #   for interest, (t1, t2) in sorted(rules, reverse=True) :
    #       print "(%d -> %d) :\t%f" % (t1, t2, interest) - freqs1[t2]/total
    #       print "(%d -> %d) :\t%f" % (t2, t1, interest) - freqs1[t1]/total

    return rules


  def get_topics_layer_from_db(self, doc_ids, min_conf_topics):
    """
    Run topic modeling for the content on the given papers and assemble the topic nodes
    and edges.
    """
    #       topics, doc_topics, tokens = topic_modeling.get_topics_online(doc_ids, ntopics=200, beta=0.1,
    #                                                                                                                               cache_folder=self.cache_folder, ign_cache=False)

    # Build topic nodes and paper-topic edges
    topic_nodes = set()
    topic_paper_edges = set()

    # Retrieve top topics for each document from the db
    topic_ids_per_doc = []
    for doc_id in doc_ids:

      topics = db.select(fields=["topic_id", "value"], table="doc_topics", where="paper_id='%s'" % doc_id)
      if len(topics):
        topic_ids, topic_values = zip(*topics)

        topic_ids_per_doc.append(topic_ids)
        #               topic_values_per_doc.append(topic_values)

        topic_nodes.update(topic_ids)
        topic_paper_edges.update([(doc_id, topic_ids[t], topic_values[t]) for t in xrange(len(topic_ids))])

      #         for d in xrange(len(doc_ids)) :
      #             topic_ids = topic_ids_per_doc[d]
      #             topic_values = topic_values_per_doc[d]


    # Normalize edge weights with the maximum value
    topic_paper_edges = normalize_edges(topic_paper_edges)

    # From the list of relevant topics f
    #       rules = self.get_frequent_topic_pairs(topic_ids_per_doc, min_conf_topics)
    topic_topic_edges = get_rules_by_lift(topic_ids_per_doc, min_conf_topics)
    topic_topic_edges = normalize_edges(topic_topic_edges)

    # Get the density of the ngram layer to feel the effect of 'min_topics_lift'
    self.topic_density = float(len(topic_topic_edges)) / len(topic_nodes)

    #       get_name = lambda u: db.select_one(fields="words", table="topic_words", where="topic_id=%d"%u)
    #       top = sorted(topic_topic_edges, key=lambda t:t[2], reverse=True)
    #       for u, v, w in top :
    #           uname = get_name(u)
    #           vname = get_name(v)
    #           print "%s\n%s\n%.3f\n" % (uname, vname, w)

    # Cast topic_nodes to list so we can assure element order
    topic_nodes = list(topic_nodes)

    return topic_nodes, topic_topic_edges, topic_paper_edges


  # def get_topics_layer(self, doc_ids, min_conf_topics) :
  #     '''
  #     Run topic modeling for the content on the given papers and assemble the topic nodes
  #     and edges.
  #     '''
  #     topics, doc_topics, tokens = topic_modeling.get_topics_online(self.cache_folder, ntopics=200,
  #                                                                                                                             beta=0.1, ign_cache=False)
  #
  #     doc_topic_above = DOC_TOPIC_THRES
  #
  #     topic_nodes = set()
  #     topic_paper_edges = set()
  #     topics_per_document = []
  #     for d in xrange(len(doc_ids)) :
  #         relevant_topics = self.get_relevant_topics(doc_topics[d], above=doc_topic_above)
  #
  #         # This data structure is needed for the correlation between topics
  #         topics_per_document.append(relevant_topics)
  #
  #         topic_nodes.update(relevant_topics)
  #         topic_paper_edges.update([(doc_ids[d], t, doc_topics[d][t]) for t in relevant_topics])
  #
  #     # Normalize edge weights with the maximum value
  #     topic_paper_edges = normalize_edges(topic_paper_edges)
  #
  #     # From the list of relevant topics f
  #     rules = self.get_frequent_topic_pairs(topics_per_document)
  #
  #     # Add only edges above certain confidence. These edge don't
  #     # need to be normalized since 0 < confidence < 1.
  #     topic_topic_edges = set()
  #     for interest, (t1, t2) in rules :
  #         if interest >= min_conf_topics :
  #             topic_topic_edges.add( (t1, t2, interest) )
  #
  #     # Cast topic_nodes to list so we can assure element order
  #     topic_nodes = list(topic_nodes)
  #
  #     # Select only the names of the topics being considered here
  #     # and store in a class attribute
  #     topic_names = topic_modeling.get_topic_names(topics, tokens)
  #     self.topic_names = {tid: topic_names[tid] for tid in topic_nodes}
  #
  #     return topic_nodes, topic_topic_edges, topic_paper_edges, tokens


  # def get_words_layer_from_db(self, doc_ids):
  #     '''
  #     Create words layers by retrieving TF-IDF values from the DB (previously calculated).
  #     '''
  #
  #     word_nodes = set()
  #     paper_word_edges = set()
  #
  #     for doc_id in doc_ids :
  #         rows = db.select(fields=["word", "value"],
  #                                          table="doc_words",
  #                                          where="paper_id='%s'"%doc_id,
  #                                          order_by=("value","desc"),
  #                                          limit=5)
  #         top_words, top_values = zip(*rows)
  #
  #         word_nodes.update(top_words)
  #         paper_word_edges.update([(doc_id, top_words[t], top_values[t]) for t in range(len(top_words))])
  #
  #     # Normalize edges weights by their biggest value
  #     paper_word_edges = normalize_edges(paper_word_edges)
  #
  #     return word_nodes, paper_word_edges


  # def get_ngrams_layer_from_db2(self, doc_ids):
  #     '''
  #     Create words layers by retrieving TF-IDF values from the DB (previously calculated).
  #     '''
  #     word_nodes = set()
  #     paper_word_edges = set()
  #
  #     ngrams_per_doc = []
  #     for doc_id in doc_ids :
  #         rows = db.select(fields=["ngram", "value"],
  #                                          table="doc_ngrams",
  #                                          where="(paper_id='%s') AND (value>=%f)" % (doc_id, config.MIN_NGRAM_TFIDF))
  #
  #
  #         if (len(rows) > 0) :
  #             top_words, top_values = zip(*rows)
  #
  #             word_nodes.update(top_words)
  #             paper_word_edges.update([(doc_id, top_words[t], top_values[t]) for t in range(len(top_words))])
  #
  #             ngrams_per_doc.append(top_words)
  #
  #     ## TEMPORARY ##
  #     # PRINT MEAN NGRAMS PER DOC
  ##        mean_ngrams = np.mean([len(ngrams) for ngrams in ngrams_per_doc])
  ##        print "%f\t" % mean_ngrams,
  #
  #     # Get get_rules_by_lift between co-occurring ngrams to create edges between ngrams
  #     word_word_edges = get_rules_by_lift(ngrams_per_doc, min_lift=config.MIN_NGRAM_LIFT)
  #
  ##        print len(word_nodes), "word nodes."
  ##        print len(word_word_edges), "word-word edges."
  ##        for e in word_word_edges :
  ##            print e
  #
  ##        for rule in sorted(rules, reverse=True) :
  ##            print rule
  #
  #     # Normalize edges weights by their biggest value
  #     word_word_edges = normalize_edges(word_word_edges)
  #     paper_word_edges = normalize_edges(paper_word_edges)
  #
  #     return word_nodes, word_word_edges, paper_word_edges


  def get_ngrams_layer_from_db(self, doc_ids, min_ngram_lift):
    """
    Create words layers by retrieving TF-IDF values from the DB (previously calculated).
    """
    word_nodes = set()
    paper_word_edges = list()

    doc_ids_str = ",".join(["'%s'" % doc_id for doc_id in doc_ids])

    MIN_NGRAM_TFIDF = 0.25

    table = "doc_ngrams"
    rows = db.select(fields=["paper_id", "ngram", "value"], table=table,
             where="paper_id IN (%s) AND (value>=%f)" % (doc_ids_str, MIN_NGRAM_TFIDF))

    #
    ngrams_per_doc = defaultdict(list)
    for doc_id, ngram, value in rows:
      word_nodes.add(ngram)
      paper_word_edges.append((str(doc_id), ngram, value))

      ngrams_per_doc[str(doc_id)].append(ngram)

    # Get get_rules_by_lift between co-occurring ngrams to create edges between ngrams
    word_word_edges = get_rules_by_lift(ngrams_per_doc.values(), min_lift=min_ngram_lift)

    # Get the density of the ngram layer to feel the effect of 'min_ngram_lift'
    self.ngram_density = float(len(word_word_edges)) / len(word_nodes)
    self.nwords = len(word_nodes)

    # Normalize edges weights by their biggest value
    word_word_edges = normalize_edges(word_word_edges)
    paper_word_edges = normalize_edges(paper_word_edges)

    return word_nodes, word_word_edges, paper_word_edges


  def get_keywords_layer_from_db(self, doc_ids, min_ngram_lift):
    """
    Create words layers by retrieving TF-IDF values from the DB (previously calculated).
    """
    word_nodes = set()
    paper_word_edges = list()

    doc_ids_str = ",".join(["'%s'" % doc_id for doc_id in doc_ids])

    where = "paper_id IN (%s)" % doc_ids_str
    if config.KEYWORDS == "extracted":
      where += " AND (extracted=1)"

    elif config.KEYWORDS == "extended":
      where += " AND (extracted=0) AND (value>=%f)" % config.MIN_NGRAM_TFIDF

    elif config.KEYWORDS == "both":
      where += " AND (value>=%f)" % config.MIN_NGRAM_TFIDF

    rows = db.select(fields=["paper_id", "keyword_name"],
             table="paper_keywords",
             where=where)

    #
    ngrams_per_doc = defaultdict(list)
    for doc_id, ngram in rows:
      word_nodes.add(ngram)
      paper_word_edges.append((str(doc_id), ngram, 1.0))

      ngrams_per_doc[str(doc_id)].append(ngram)

    # Get get_rules_by_lift between co-occurring ngrams to create edges between ngrams
    word_word_edges = get_rules_by_lift(ngrams_per_doc.values(), min_lift=min_ngram_lift)

    # Get the density of the ngram layer to feel the effect of 'min_ngram_lift'
    self.ngram_density = float(len(word_word_edges)) / len(word_nodes)
    self.nwords = len(word_nodes)

    # Normalize edges weights by their biggest value
    word_word_edges = normalize_edges(word_word_edges)
    paper_word_edges = normalize_edges(paper_word_edges)

    return word_nodes, word_word_edges, paper_word_edges


  def get_papers_atts(self, papers):
    """
    Fetch attributes for each paper from the DB.
    """
    atts = {}
    for paper in papers:
      title, jornal, conf = db.select_one(["normal_title", "jornal_id", "conf_id"], table="papers", where="id='%s'" % paper)
      title = title if title else ""
      venue = jornal if jornal else conf
      atts[paper] = {"label": title, "title": title, "venue": venue}

    return atts


  def get_authors_atts(self, authors):
    """
    Fetch attributes for each author from the DB.
    """
    atts = {}
    for author in authors:
      name, email, affil = db.select_one(["name", "email", "affil"], table="authors", where="cluster=%d" % author)
      npapers = str(db.select_one("count(*)", table="authors", where="cluster=%d" % author))
      name = name if name else ""
      email = email if email else ""
      affil = affil if affil else ""

      atts[author] = {"label": name, "name": name, "email": email, "affil": affil, "npapers": npapers}

    return atts


  def get_topics_atts(self, topics):
    """
    Fetch attributes for each topic.
    """
    topic_names = db.select(fields="words", table="topic_words", order_by="topic_id")
    atts = {}
    for topic in topics:
      topic_name = topic_names[topic]
      atts[topic] = {"label": topic_name, "description": topic_name}

    return atts


  def get_words_atts(self, words):
    """
    Fetch attributes for each word.
    """
    atts = {}
    for word in words:
      atts[word] = {"label": word}

    return atts


  def assemble_layers(self, pubs, citation_edges,
            authors, coauth_edges, auth_edges,
            # topics, topic_topic_edges, paper_topic_edges,
            # ngrams, ngram_ngram_edges, paper_ngram_edges,
            # venues, pub_venue_edges,
            affils, author_affil_edges, affil_affil_edges):
    """
    Assembles the layers as an unified graph. Each node as an unique id, its type (paper,
    author, etc.) and a readable label (paper title, author name, etc.)
    """
    graph = nx.DiGraph()

    # These map the original identifiers for each type (paper doi, author id,
    # etc.) to the new unique nodes id.
    pubs_ids = {}
    authors_ids = {}
    # topics_ids = {}
    words_ids = {}
    venues_ids = {}
    affils_ids = {}

    # Controls the unique incremental id generation
    next_id = 0

    # Add each paper providing an unique node id. Some attributes must be added
    # even if include_attributes is True, since they are used in ranking algorithm.
    for pub in pubs:
      pub = str(pub)

      #         if hasattr(self, 'query_sims') :
      #             query_score = float(self.query_sims[paper])  #if paper in self.query_sims else 0.0
      #         else :
      #             query_score = 0.0

      graph.add_node(next_id,
               type="paper",
               entity_id=pub,
               year=self.pub_years[pub]
               )

      pubs_ids[pub] = next_id
      next_id += 1

    # Add citation edges (directed)
    for paper1, paper2, weight in citation_edges:
      graph.add_edge(pubs_ids[paper1], pubs_ids[paper2], weight=weight)
      # graph.add_edge(pubs_ids[paper2], pubs_ids[paper1], weight=weight) # try undirected


    # Add each author providing an unique node id
    for author in authors:
      graph.add_node(next_id, type="author", entity_id=author)

      authors_ids[author] = next_id
      next_id += 1


    # Add co-authorship edges on both directions (undirected)
    for author1, author2, weight in coauth_edges:
      graph.add_edge(authors_ids[author1], authors_ids[author2], weight=weight)
      graph.add_edge(authors_ids[author2], authors_ids[author1], weight=weight)

    # Add authorship edges on both directions (undirected)
    for paper, author in auth_edges:
      graph.add_edge(pubs_ids[paper], authors_ids[author], weight=1.0)
      graph.add_edge(authors_ids[author], pubs_ids[paper], weight=1.0)


    ####################################

    #       # Add topic nodes
    #       for topic in topics :
    #           graph.add_node(next_id, type="topic", entity_id=topic)
    #
    #           topics_ids[topic] = next_id
    #           next_id += 1
    #
    #       # Add topic correlation edges (directed)
    #       for topic1, topic2, weight in topic_topic_edges :
    #           graph.add_edge(topics_ids[topic1], topics_ids[topic2], weight=weight)
    #           graph.add_edge(topics_ids[topic2], topics_ids[topic1], weight=weight)
    #
    #       # Add paper-topic edges (directed)
    #       for paper, topic, weight in paper_topic_edges :
    #           graph.add_edge(pubs_ids[paper], topics_ids[topic], weight=weight)
    #           graph.add_edge(topics_ids[topic], pubs_ids[paper], weight=weight)

    ####################################
    # # Add ngram nodes
    # for ngram in ngrams:
    #   graph.add_node(next_id, type="keyword", entity_id=ngram)

    #   words_ids[ngram] = next_id
    #   next_id += 1

    # #        Add word-word edges (undirected)
    # for w1, w2, weight in ngram_ngram_edges:
    #   graph.add_edge(words_ids[w1], words_ids[w2], weight=weight)
    #   graph.add_edge(words_ids[w2], words_ids[w1], weight=weight)

    # # Add paper-word edges (undirected)
    # for paper, word, weight in paper_ngram_edges:
    #   graph.add_edge(pubs_ids[paper], words_ids[word], weight=weight)
    #   graph.add_edge(words_ids[word], pubs_ids[paper], weight=weight)

    ####################################
    # Add venues to the graph
    # for venue in venues:
    #   graph.add_node(next_id, type="venue", entity_id=venue)

    #   venues_ids[venue] = next_id
    #   next_id += 1

    # for pub, venue, weight in pub_venue_edges:
    #   graph.add_edge(pubs_ids[pub], venues_ids[venue], weight=weight)
    #   graph.add_edge(venues_ids[venue], pubs_ids[pub], weight=weight)


  # Add affils to the graph
    for affil in affils:
      graph.add_node(next_id, type="affil", entity_id=affil)

      affils_ids[affil] = next_id
      next_id += 1

    # author_affils_dict = defaultdict()
    for author, affil, weight in author_affil_edges:
      graph.add_edge(authors_ids[author], affils_ids[affil], weight=weight)
      graph.add_edge(affils_ids[affil], authors_ids[author], weight=weight)
      # try:
      #   author_affils_dict[author].add(affil)
      # except:
      #   author_affils_dict[author] = set([affil])

    # try affil-affil layer
    # 1)
    # for affil_1, affil_2, weight in affil_affil_edges:
    #   graph.add_edge(affils_ids[affil_1], affils_ids[affil_2], weight=weight)
    #   graph.add_edge(affils_ids[affil_2], affils_ids[affil_1], weight=weight)

    # 2)

    # for author1, author2, weight in coauth_edges:
    #   if author_affils_dict.has_key(author1) and author_affils_dict.has_key(author2):
    #     for affil1 in author_affils_dict[author1]:
    #       for affil2 in author_affils_dict[author2]:
    #         if affil1 != affil2:
    #           graph.add_edge(affils_ids[affil1], affils_ids[affil2], weight=weight)
    #           graph.add_edge(affils_ids[affil2], affils_ids[affil1], weight=weight)

    # Get the attributes for each author
    # Get attributes for each paper
    if self.include_attributes:
      # add_attributes(graph, pubs, pubs_ids, self.get_papers_atts(pubs))
      # add_attributes(graph, authors, authors_ids, self.get_authors_atts(authors))
      # add_attributes(graph, topics, topics_ids, self.get_topics_atts(topics))
      # add_attributes(graph, words, words_ids, self.get_words_atts(words))
      pass
    return graph


  def parse_tfidf_line(self, line):
    parts = line.strip().split()
    tokens = parts[0::2]
    tfidf = map(float, parts[1::2])
    return dict(zip(tokens, tfidf))


  def get_edge_contexts(self, papers, citation_edges):

    citation_edges = set(citation_edges)

    tokens_per_citation = {}
    for citing in papers:
      if os.path.exists(config.CTX_PATH % citing):
        with open(config.CTX_PATH % citing, "r") as file:
          for line in file:
            cited, tokens_tfidf = line.strip().split('\t')

            if (citing, cited) in citation_edges:
              tokens_per_citation[(citing, cited)] = self.parse_tfidf_line(tokens_tfidf)

    return tokens_per_citation


  def get_venues_layer(self, papers):
    """
    Returns the venues' ids and edges from publications to venues according
    to the venues used in the publications.
    """
    if hasattr(papers, '__iter__'):
      if len(papers) == 0:
        return [], []
      else:
        paper_str = ",".join(["'%s'" % paper_id for paper_id in papers])
    else:
      raise TypeError("Parameter 'papers' is of unsupported type. Iterable needed.")

    venues = set()
    pub_venue_edges = list()
    rows = db.select(fields=["id", "jornal_id", "conf_id"], table="papers", \
            where="id IN (%s)"%paper_str)
    for pub, jornal_id, conf_id in rows:
      if jornal_id:
        venues.add(jornal_id)
        pub_venue_edges.append((pub, jornal_id, 1.0))
      elif conf_id:
        venues.add(conf_id)
        pub_venue_edges.append((pub, conf_id, 1.0))

    return list(venues), pub_venue_edges


  def get_affils_layer(self, authors, related_papers):
    """
    Returns the affils' ids and edges from authors to affiliations according
    to authors.
    """
    if hasattr(authors, '__iter__'):
      if len(authors) == 0:
        return [], []
      else:
        author_str = ",".join(["'%s'" % author_id for author_id in authors])
    else:
      raise TypeError("Parameter 'authors' is of unsupported type. Iterable needed.")

    if hasattr(related_papers, '__iter__'):
      if len(related_papers) == 0:
        return [], []
      else:
        paper_str = ",".join(["'%s'" % paper_id for paper_id in related_papers])
    else:
      raise TypeError("Parameter 'related_papers' is of unsupported type. Iterable needed.")


    affils = set()
    author_affil_edges = set()
    affil_affil_edges = set()
    rows = db.select(["paper_id", "author_id", "affil_id"], "paper_author_affils",\
           where="author_id IN (%s) and paper_id IN (%s)"%(author_str, paper_str))


    missing_count = 0
    missing_author = 0
    hit_count = 0
    author_affils = defaultdict()
    for paper_id, author_id, affil_id in rows:
      if affil_id == '':
        missing_count += 1
      try:
        author_affils[author_id].add(affil_id)
      except:
        author_affils[author_id] = set([affil_id])

    for author_id, affil_ids in author_affils.iteritems():
      # add affil-affil edges
      # for i in range(len(affil_ids)-1):
      #   if list(affil_ids)[i] != '':
      #     for j in range(i+1,len(affil_ids)):
      #       if list(affil_ids)[j] != '':
      #         affil_affil_edges.add((list(affil_ids)[i], list(affil_ids)[j], 1.0))


      for each in affil_ids:
        if each != '':
          affils.add(each)
          author_affil_edges.add((author_id, each, 1.0))
        else:
          # To be improved, we only retrieve affils
          # when we don't know any affils which the author belongs to
          if len(affil_ids) == 1:
            missing_author += 1
            # we check external data (e.g., csx dataset) and do string matching which is knotty.
            match_affil_ids = retrieve_affils_by_authors(author_id)
            if match_affil_ids:
              hit_count += 1
            for each_affil in match_affil_ids:
              affils.add(each_affil)
              author_affil_edges.add((author_id, each_affil, 1.0))

    print "missing count: %s"%missing_count
    print "missing author: %s"%missing_author
    print "hit count: %s"%hit_count

    # # count = 0
    # tmp_authors = set()
    # for paper_id, author_id, affil_id in rows:
    #   # Since coverage of affils in MAG dataset is quite low,
    #   # we use data from CSX dataset as a complement.
    #   if affil_id == '':
    #     try:
    #       if author_id in tmp_authors: # For the sake of simplicity, we don't look up affils in this case.
    #         continue

    #       # count += 1
    #       tmp_authors.add(author_id)
    #       # import pdb;pdb.set_trace()

    #       # 1) first, we check paper_author_affils table.
    #       if author_id in rows
    #       # 2) if 1) fails, then we check csx_authors table and do string matching which is knotty.
    #       match_affil_ids = retrieve_affils_by_authors(author_id)

    #       if match_affil_ids:
    #         affils.update(match_affil_ids)
    #         for each in match_affil_ids:
    #           author_affil_edges.add((author_id, each, 1.0))

    #     except Exception, e:
    #       print e
    #       continue

    #   else:
    #     affils.add(affil_id)
    #     author_affil_edges.add((author_id, affil_id, 1.0))


    print len(affils), len(author_affil_edges), len(affil_affil_edges)
    return list(affils), list(author_affil_edges), list(affil_affil_edges)


  def build(self, conf_name, year, n_hops, min_topic_lift, min_ngram_lift, exclude=[]):
    """
    Build graph model from given conference.
    """

    log.debug("Building model for conference='%s' and hops=%d." % (conf_name, n_hops))

    pubs, citation_edges = self.get_pubs_layer(conf_name, year, n_hops, set(exclude))
    log.debug("%d pubs and %d citation edges." % (len(pubs), len(citation_edges)))

    authors, coauth_edges, auth_edges = self.get_authors_layer(pubs)
    log.debug("%d authors, %d co-authorship edges and %d authorship edges." % (
      len(authors), len(coauth_edges), len(auth_edges)))

    #       topics, topic_topic_edges, pub_topic_edges = self.get_topics_layer_from_db(pubs, min_topic_lift)
    #       log.debug("%d topics, %d topic-topic edges and %d pub-topic edges."
    #                                       % (len(topics), len(topic_topic_edges), len(pub_topic_edges)))

    # # Use the standard ngrams formulation if the config says so
    # if config.KEYWORDS == "ngrams":
    #   words, word_word_edges, pub_word_edges = self.get_ngrams_layer_from_db(pubs, min_ngram_lift)

    # # Otherwise use some variant of a keywords' layer
    # else:
    #   words, word_word_edges, pub_word_edges = self.get_keywords_layer_from_db(pubs, min_ngram_lift)
    # log.debug("%d words and %d pub-word edges." % (len(words), len(pub_word_edges)))

    # venues, pub_venue_edges = self.get_venues_layer(pubs)
    # log.debug("%d venues and %d pub-venue edges." % (len(venues), len(pub_venue_edges)))

    affils, author_affil_edges, get_affils_layer = self.get_affils_layer(authors, pubs)
    log.debug("%d affiliations and %d pub-affil edges." % (len(affils), len(author_affil_edges)))

    graph = self.assemble_layers(pubs, citation_edges,
                   authors, coauth_edges, auth_edges,
                   # None, None, None,
                   #                                                        topics, topic_topic_edges, pub_topic_edges,
                   # words, word_word_edges, pub_word_edges,
                   # venues, pub_venue_edges,
                   affils, author_affil_edges, get_affils_layer)

    # Writes the contexts of each edge into a file to be used efficiently
    # on the ranking algorithm.
    #       self.write_edge_contexts(papers, citation_edges, ctxs_file)

    # Writes the gexf
    #       write_graph(graph, model_file)
    return graph



if __name__ == '__main__':
  log.basicConfig(format='%(asctime)s [%(levelname)s] : %(message)s', level=log.INFO)
  mb = ModelBuilder()
  graph = mb.build_full_graph()

