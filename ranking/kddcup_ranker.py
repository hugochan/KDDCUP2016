'''
Created on Mar 15, 2016

@author: hugo
'''

import os
import sys
import numpy as np
import networkx as nx
import logging as log
import logging
from _collections import defaultdict
from networkx.linalg.graphmatrix import adjacency_matrix
from mymysql.mymysql import MyMySQL
import random
from networkx.classes.digraph import DiGraph
from networkx.generators.stochastic import stochastic_graph
import utils
from config import DATA
import cPickle
import time
from networkx.algorithms.centrality.katz import katz_centrality


# params=                           {  'papers_relev': .5, # .5
#                               'authors_relev': .5, # .5
#                               'author_affils_relev': .3, # .3
# }

def pagerank(G, alpha=0.85, pers=None, max_iter=100,
                         tol=1.0e-8, nstart=None, weight='weight', node_types=None):
    """Return the PageRank of the nodes in the graph.

    PageRank computes a ranking of the nodes in the graph G based on
    the structure of the incoming links. It was originally designed as
    an algorithm to rank web pages.

    Parameters
    -----------
    G : graph
        A NetworkX graph

    alpha : float, optional
        Damping parameter for PageRank, default=0.85

    pers: dict, optional
         The "pers vector" consisting of a dictionary with a
         key for every graph node and nonzero pers value for each node.

    max_iter : integer, optional
        Maximum number of iterations in power method eigenvalue solver.

    tol : float, optional
        Error tolerance used to check convergence in power method solver.

    nstart : dictionary, optional
        Starting value of PageRank iteration for each node.

    weight : key, optional
        Edge data key to use as weight. If None weights are set to 1.

    Returns
    -------
    pagerank : dictionary
         Dictionary of nodes with PageRank as value

    Notes
    -----
    The eigenvector calculation is done by the power iteration method
    and has no guarantee of convergence.    The iteration will stop
    after max_iter iterations or an error tolerance of
    number_of_nodes(G)*tol has been reached.
    """

    if len(G) == 0:
            return {}

    # create a copy in (right) stochastic form
    W=nx.stochastic_graph(G, weight=weight)
#   W = G
    scale=1.0/W.number_of_nodes()

    # choose fixed starting vector if not given
    if nstart is None:
            x=dict.fromkeys(W,scale)
    else:
            x=nstart
            # normalize starting vector to 1
            s=1.0/sum(x.values())
            for k in x: x[k]*=s

    # assign uniform pers vector if not given
    if pers is None:
            pers=dict.fromkeys(W,scale)
    else:
            # Normalize the sum to 1
            s=sum(pers.values())
#           p=pers/sum(pers)
            for k in pers.keys():
                pers[k] /= s

            if len(pers)!=len(G):
                    raise Exception('Personalization vector must have a value for every node')


    # "dangling" nodes, no links out from them
    out_degree=W.out_degree()
    dangle=[n for n in W if out_degree[n]==0.0]

    i=0
    while True: # power iteration: make up to max_iter iterations
            xlast=x
            x=dict.fromkeys(xlast.keys(), 0)

            # "dangling" nodes only consume energies, so we release these energies manually
            danglesum=alpha*scale*sum(xlast[n] for n in dangle)
            # danglesum = 0

            for n in x:
                    # this matrix multiply looks odd because it is
                    # doing a left multiply x^T=xlast^T*W
                    for nbr in W[n]:
                            x[nbr] += alpha*xlast[n]*W[n][nbr][weight]

#                           c[nbr] += 1
#                           if node_types :
#                               l[nbr][node_types[n]]+= dx
#                           if node_types[nbr]==0 :
#                               print node_types[nbr], dx
#                               print

                    # x[n] += danglesum
                    # if G.node[n]['type'] == 'affil':
                    #     for kk, vv in pers.items():
                    #         if G.node[kk]['type'] == 'affil':
                    #             x[n] += (1-params['author_affils_relev'])*(1-alpha)*vv*xlast[kk]
                    #         elif G.node[kk]['type'] == 'author':
                    #             x[n] += params['author_affils_relev']*(1-alpha)*vv*xlast[kk]
                    # elif G.node[n]['type'] == 'author':
                    #     for kk, vv in pers.items():
                    #         if G.node[kk]['type'] == 'author':
                    #             x[n] += (1-params['author_affils_relev']-params['authors_relev'])*(1-alpha)*vv*xlast[kk]

                    #         elif G.node[kk]['type'] == 'affil':
                    #             x[n] += params['author_affils_relev']*(1-alpha)*vv*xlast[kk]

                    #         elif G.node[kk]['type'] == 'paper':
                    #             x[n] += params['authors_relev']*(1-alpha)*vv*xlast[kk]
                    # elif G.node[n]['type'] == 'paper':
                    #     for kk, vv in pers.items():
                    #         if G.node[kk]['type'] == 'paper':
                    #             x[n] += params['papers_relev']*(1-alpha)*vv*xlast[kk]

                    #         elif G.node[kk]['type'] == 'author':

                    #             x[n] += params['authors_relev']*(1-alpha)*vv*xlast[kk]

                    x[n]+=danglesum+(1-alpha)*pers[n]
                    # x[n]+=danglesum+(1-alpha)*pers[n]
#                   l[n][4]+=danglesum+(1.0-alpha)*pers[n]

            # normalize vector
            s=1.0/sum(x.values())
            for n in x:
                    x[n]*=s
#                   l[n]*=s

#           print c[637], ' '.join(map(str,np.round(100*l[637],3))), "\t", \
#                       c[296], ' '.join(map(str,np.round(100*l[296],3)))

            # check convergence, l1 norm
            err=sum([abs(x[n]-xlast[n]) for n in x])
            if err < tol:
                    break
            if i>max_iter:
                    raise Exception('pagerank: power iteration failed to converge '
                                                            'in %d iterations.'%(i-1))
            i+=1

    # Returns:
    #   x: PageRank of each node;
    #   l: Detailed contributions of each layer;
    #   i: Iterations to converge.
    return x, i


#def remove_nodes(graph, type):
#   for node in graph.nodes() :
#       if graph.node[node]["type"]==type :
#           graph.remove_node(node)


def rank_nodes(graph, papers_relev=0.2,
                                         authors_relev=0.2,
                                         # topics_relev=0.2,
                                         words_relev=0.2,
                                         venues_relev=0.2,
                                         author_affils_relev=0.2,
                                         age_relev=0.5,
                                         # query_relev=0.5,
                                         # ctx_relev=0.5,
                                         alpha=0.3,
                                         # query_telep=True,
                                         init_pg=True,
                                         out_file=None,
                                         stats_file=None,
                                         **kwargs) :

    # If 'graph' is a string then a path was provided, so we load the graph from it
    if (isinstance(graph, basestring)) :
        graph = nx.read_gexf(graph, node_type=int)


    # Truncate parameters between 0 and 1
#   age_relev = max(min(age_relev, 1.0), 0.0)
#   query_relev = max(min(query_relev, 1.0), 0.0)

    # Layer relevance parameters are exponentiate to increase sensitivity and normalized to sum to 1
#   rho = np.exp([papers_relev, authors_relev, topics_relev, words_relev])
    rho = np.asarray([papers_relev, authors_relev, author_affils_relev])
    # rho = np.asarray([papers_relev, authors_relev, words_relev, author_affils_relev])
    # rho = np.asarray([papers_relev, authors_relev, words_relev, venues_relev, author_affils_relev])

    # Each row and col sumps up to 1
    rho_papers = rho[0] / (rho[0] + rho[1])
    rho_authors = rho[1] / (rho[0] + rho[1])
    rho_affils = rho[2]

    # rho_papers, rho_authors, rho_affils = rho/rho.sum()
    # rho_papers, rho_authors, rho_words, rho_affils = rho/rho.sum()
    # rho_papers, rho_authors, rho_words, rho_venues, rho_affils = rho/rho.sum()

    # log.debug("Transitions paper -> x: paper=%.3f, author=%.3f, words=%.3f, venues=%.3f" %
    #                                     (rho_papers, rho_authors, rho_words, rho_venues))

    # Transition probabilities between layers. The rows and columns correspond to
    # the papers, authors, topics and words layers. So for example, the value at
    # (i,j) is the probability of the random walker to go from layer i to layer j.
    rho = np.array([[rho_papers,     rho_authors,              0],
                [rho_authors,  1.0-rho_authors-rho_affils,             rho_affils],
                [         0,       1.0,                 0]])
    # rho = np.array([[rho_papers,     rho_authors,    rho_words,          0],
    #             [rho_authors,  1.0-rho_authors-rho_affils,    0,         rho_affils],
    #             [rho_words,                  0,   1.0-rho_words,              0],
    #             [         0,       rho_affils,          0,     0,      1.0-rho_affils]])
    # rho = np.array([[rho_papers,     rho_authors,    rho_words,      rho_venues,    0],
    #             [rho_authors,  1.0-rho_authors-rho_affils,    0,     0,    rho_affils],
    #             [rho_words,                  0,   1.0-rho_words,              0,    0],
    #             [rho_venues,                0,               0,  1.0-rho_venues,    0],
    #             [         0,       rho_affils,          0,     0,      1.0-rho_affils]])
    print rho
    # Maps the layers name to the dimensions
    # layers = {"paper":0, "author":1, "keyword":2, "venue":3, "affil":4}
    # layers = {"paper":0, "author":1, "keyword":2, "affil":3}
    layers = {"paper":0, "author":1, "affil":2}

    # Alias vector to map nodes into their types (paper, author, etc.) already
    # as their numeric representation (paper=0, author=1, etc.) as listed above.

    node_types = {u: layers[graph.node[u]["type"]] for u in graph.nodes()}
    # Quick alias method to check if the node is paper or affil
    is_paper = lambda n: (node_types[n] == layers["paper"])
    is_affil = lambda n: (node_types[n] == layers["affil"])
    is_author = lambda n: (node_types[n] == layers["author"])

#   print graph.number_of_nodes(),

    # Remove isolated nodes
    graph.remove_nodes_from(nx.isolates(graph))

#   print graph.number_of_nodes()

    # Assemble our personalization vector according to similarity to the query provided.
    # Only paper nodes get teleported to, so other layers get 0 as factors.
    # Get year median to replace missing values.
    # old_year = 1950
    # current_year = 2016
    # npapers = 0
    # years = []
    # for u in graph.nodes() :
    #     if is_paper(u) :
    #         npapers += 1

    #         if (graph.node[u]["year"] > 0) :
    #             years.append(graph.node[u]["year"])

    # year_median = np.median(years)

    # log.debug("Using year=%d (median) for missing values." % int(year_median))


    # Normalize weights within each kind of layer transition, e.g., normalize papers to
    # topic edges separately from papers to papers edges.
    for u in graph.nodes() :

        weights = defaultdict(float)
        out_edges = graph.out_edges(u, data=True)
        for u, v, atts in out_edges:

        # Also apply the age attenuator to control relevance of old and highly cited papers
            if is_paper(v) and is_paper(u):
                # year = graph.node[v]["year"]
                # if year == 0:
                #     year = year_median
                # else:
                #     year = min(max(year, old_year), current_year)

                weight = 1.0
                # weight *= np.exp(-(age_relev)*(current_year-year))

                atts['weight'] = weight
#               print weight


            # Sum total output weight of current node (u) to each layer separately.
            weights[node_types[v]] = max(atts['weight'], weights[node_types[v]])

        # Here, beside dividing by the total weight for the type of transition, we
        # multiply by the probability of the transition, as given by the rho matrix.
        for u, v, atts in out_edges:
            from_layer = node_types[u]
            to_layer   = node_types[v]

#           if (weights[to_layer]==0) :
#               print
            atts['weight'] *= rho[from_layer][to_layer]/weights[to_layer]

    # Create personalization dict. The probability to leap to a publication node
    # is proportional to the similarity of that publication's text to the query.
    # Other nodes are not leaped to. If all query scores are 0, we just use an
    # uniform probability. The parameter 'query_relev' controls how much of this
    # query weighting is applied.

    # We do not have query relevance in this task.

    # # option 1) random jump on affils
    # naffils = 0
    # for u in graph.nodes():
    #     if is_affil(u):
    #         naffils += 1


    # uniform_pers = 1.0/naffils
    # pers = {node: uniform_pers*is_affil(node) for node in graph.nodes()} # try random jump on affils

    # # option 2) random jump on affils and authors, doesn't improve results compared to 1)
    # nn = 0
    # for u in graph.nodes():
    #     if is_affil(u) or is_author(u):
    #         nn += 1

    # uniform_pers = 1.0/nn
    # pers = {node: uniform_pers*(is_affil(node) or is_author(node)) for node in graph.nodes()} # try random jump on affils or authors



    # option 3) random jump on in the whole network, doesn't improve results compared to 1)
    uniform_pers = 1.0/len(graph.nodes())
    pers = {node: uniform_pers for node in graph.nodes()} # try random jump on affils or authors


   # # option 4) random jump on separated multilayers
   #  naffils = 0
   #  nauthors = 0
   #  npapers = 0
   #  for u in graph.nodes():
   #      if is_affil(u):
   #          naffils += 1
   #      if is_author(u):
   #          nauthors += 1
   #      if is_paper(u):
   #          npapers += 1

   #  uniform_affil_pers = 1.0/naffils
   #  uniform_author_pers = 1.0/nauthors
   #  uniform_paper_pers = 1.0/npapers

   #  pers = defaultdict()
   #  for node in graph.nodes():
   #      if is_affil(node):
   #          pers[node] = uniform_affil_pers
   #      if is_author(node):
   #          pers[node] = uniform_author_pers
   #      if is_paper(node):
   #          pers[node] = uniform_paper_pers


   # # option 5) random jump on separated multilayers
   #  naffils = 0
   #  nauthors = 0
   #  npapers = 0
   #  for u in graph.nodes():
   #      if is_affil(u):
   #          naffils += 1
   #      if is_author(u):
   #          nauthors += 1
   #      if is_paper(u):
   #          npapers += 1

   #  uniform_affil_pers = 1.0 / (naffils + nauthors)
   #  uniform_author_pers = 1.0 / (nauthors + npapers + naffils)
   #  uniform_paper_pers = 1.0/ (npapers + nauthors)

   #  pers = defaultdict()
   #  for node in graph.nodes():
   #      if is_affil(node):
   #          pers[node] = uniform_affil_pers
   #      if is_author(node):
   #          pers[node] = uniform_author_pers
   #      if is_paper(node):
   #          pers[node] = uniform_paper_pers


    # Run page rank on the constructed graph
    scores, _niters = pagerank(graph, alpha=(1.0-alpha), pers=pers, node_types=node_types, max_iter=10000)

#   if stats_file :
#       with open(stats_file, "a") as f :
#           print >> f, "%d\t%f" % (niters, e-s)

    # Write a slight modified graph to file (normalized edges and rank
    # value included as attribute)
#   if out_file :
#       for id, relevance in pg.items() :
#           graph.add_node(id, relevance=float(relevance))
#
#       nx.write_gexf(graph, out_file, encoding="utf-8")

    # If needed, dump all the PG values to be used as init values
    # in future computation to speedup convergence.
#   if True :
#       key_names = {0:"paper_id", 1:"author_id", 2:"topic_id", 3:"word_id"}
#       nstart_values = {}
#       for n, score in rank :
#           nstart_values[(node_types[n], graph.node[n][key_names[node_types[n]]])] = score
#
#       with open(DATA + "cache/nstart.pg", "w") as nstart_file :
#           cPickle.dump(nstart_values, nstart_file)

    return scores


def rank_nodes_baselines(graph, method="katz", limit=20) :

    # If 'graph' is a string then a path was provided, so we load the graph from it
    if (isinstance(graph, basestring)) :
        graph = nx.read_gexf(graph, node_type=int)

    if method=="katz" :
        r = katz_centrality(graph, alpha=0.01, beta=1.0)
    elif method=="hits_hub" :
        hubs, auth = nx.hits(graph, max_iter=500)
        r = hubs
    elif method=="hits_auth" :
        hubs, auth = nx.hits(graph, max_iter=500)
        r = auth
    else :
        raise ValueError("Invalid method parameter: '%s'" % method)


    rank = sorted(r.items(), key=lambda (k,v):v, reverse=True)

    results = []
    for node_id, score in rank :
        if graph.node[node_id]["type"]=="paper" :
            results.append((node_id, graph.node[node_id]["paper_id"], score))

        if len(results) == limit :
            break

    return results


def test_attenuators() :
    import matplotlib.pyplot as pp

    # Query score attenuator
    o = 5
    q = lambda x: np.exp(-5*o*(1-x))

    x = np.linspace(0.0, 1.0, 100)
    y = q(x)

    pp.ylim(0,1.05)
    pp.plot(x, y, lw=1.5)
    pp.show()

    # Age attenuator
#   f = lambda x: np.exp(-5*o*(1-x))




if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s [%(levelname)s] : %(message)s', level=logging.INFO)

#   test_attenuators()
#   sys.exit()

#   g = DiGraph()
#   g.add_edge(0,1,weight=1.0)
#   g.add_edge(1,2,weight=5.0)
#   g.add_edge(1,0,weight=5.0)
#   print adjacency_matrix(stochastic_graph(g))

#   query = "subspace+clustering_N100_H1"
    query = "subgraph+mining"
#   query = "data+cleaning_N100_H1"
#   query = "image+descriptor_N100_H1"

    graph = nx.read_gexf("models/%s.gexf" % query, node_type=int)

#   print "The Dense", len(graph.in_edges(637)), \
#                                           sum([a["weight"] for u,v,a in graph.in_edges(637, data=True)]), \
#                                           np.mean([graph.out_degree(u) for u,v in graph.in_edges(637)])
#
#   print "GSpan", len(graph.in_edges(296)), \
#                                   sum([a["weight"] for u,v,a in graph.in_edges(296, data=True)]), \
#                                   np.mean([graph.out_degree(u) for u,v in graph.in_edges(296)])
#   sys.exit()

    rank = rank_nodes(graph, 1.0, 1.0, 1.0, 1.0, ctx_relev=0.5, query_relev=0.5, age_relev=0.5,
                                                limit=15, out_file="graphs/ranks/%s.gexf" % query)

    print
    for node_id, paper_id, query_score, score, score_layers in rank :
        print "{%15s,  %4d,  %3d,  %.4f} : [%.2f]   %-70s  |  %s" % (paper_id,
                                                                                             graph.node[node_id]["year"],
                                                                                             len(graph.in_edges(node_id)),
                                                                                             100*query_score,
                                                                                             100*score,
                                                                                             utils.get_title(paper_id)[:70],
                                                                                             ' '.join(map(str,np.round(100*score_layers,3))))

