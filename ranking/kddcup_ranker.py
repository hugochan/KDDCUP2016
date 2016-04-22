'''
Created on Mar 15, 2016

@author: hugo
'''

import os
import sys
import numpy as np
from scipy import optimize
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
from config import DATA, PARAMS
import cPickle
import time
import itertools
from networkx.algorithms.centrality.katz import katz_centrality

old_settings = np.seterr(all='warn', over='raise')

# params=                           {  'papers_relev': .5, # .5
#                               'authors_relev': .5, # .5
#                               'author_affils_relev': .3, # .3
# }



# Metropolis Hastings
def pagerank2(G, alpha=0.85, pers=None, max_iter=100,
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
    out_degree = W.out_degree()
    # import pdb;pdb.set_trace()
    for node in W:
        for k, v in W[node].iteritems():
            v[weight] *= min(1, float(out_degree[node])/out_degree[k] if out_degree[k] else 1)
        W[node][node] = {weight: 1 - sum([y for x in W[node].values() for y in x.values()])}



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
                        try:
                            x[nbr] += alpha*xlast[n]*W[n][nbr][weight]
                        except:
                            import pdb;pdb.set_trace()
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
                    # x[n]+=danglesum+(1-alpha)*np.array(pers.values()).dot(np.array(xlast.values()))
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
                    # x[n]+=danglesum+(1-alpha)*np.array(pers.values()).dot(np.array(xlast.values()))
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



def get_delta(authors, author_affils, affil_score):
    global global_authoridx, global_affilidx

    nauthors = len(authors)
    Delta = np.zeros((nauthors, nauthors))

    for author1 in authors:
        b = sum([affil_score[global_affilidx[x]] for x in author_affils[author1]]) if author1 in author_affils else 0.0
        for author2 in authors:
            if author1 == author2:    continue

            a = sum([affil_score[global_affilidx[x]] for x in author_affils[author2]]) if author2 in author_affils else 0.0
            Delta[global_authoridx[author1]][global_authoridx[author2]] = a - b

    return Delta


global global_authoridx, global_affilidx, global_beta, global_authors, global_affils, global_author_authors, global_author_affils
gama = 1.0

def object_func(affil_score, sign=1):
    """
    object function
    """
    global global_authoridx, global_beta, global_authors, global_author_authors, global_author_affils

    Delta = get_delta(global_authors, global_author_affils, affil_score)

    obj_func = 0

    with np.errstate(divide='raise'):
        try:
            # prob = lambda (author1, author2, cond): np.log(1.0/( 1.0 + np.exp(-global_beta * Delta[global_authoridx[author1]][global_authoridx[author2]]))) \
            #     if cond else np.log(1.0 - 1.0/( 1.0 + np.exp(-global_beta * Delta[global_authoridx[author1]][global_authoridx[author2]])))

            prob = lambda (author1, author2, cond): global_author_authors[author1][author2] * global_beta * Delta[global_authoridx[author1]][global_authoridx[author2]] - np.log(1.0 + np.exp(global_beta * Delta[global_authoridx[author1]][global_authoridx[author2]])) \
                            if cond else -1 * gama * np.log(1.0 + np.exp(global_beta * Delta[global_authoridx[author1]][global_authoridx[author2]]))

            # obj_func = sum([prob((author1, author2, author1 in global_author_authors and author2 in global_author_authors[author1])) \
            #     for author1 in global_authors for author2 in global_authors if not author1 == author2])
            for author1 in global_authors:
                for author2 in global_authors:
                    if author1 == author2:
                        continue

                    # weighted sum
                    obj_func += prob((author1, author2, author1 in global_author_authors and author2 in global_author_authors[author1]))

        except Exception, e:
            print e
            import pdb;pdb.set_trace()


    return sign * obj_func


def fprime(affil_score, sign=1):
    """
    gradient of object function
    """

    global global_authoridx, global_affilidx, global_beta, global_authors, global_affils, global_author_authors, global_author_affils
    Delta = get_delta(global_authors, global_author_affils, affil_score)

    # gradient
    delta_prob = lambda (author1, author2, affil, cond): global_author_authors[author1][author2] * global_beta * ((1.0 if author2 in global_author_affils and affil in global_author_affils[author2] else 0.0) - (1.0 if author1 in global_author_affils and affil in global_author_affils[author1] else 0.0)) \
            * np.exp(-global_beta * Delta[global_authoridx[author1]][global_authoridx[author2]]) / ( 1.0 + np.exp(-global_beta * Delta[global_authoridx[author1]][global_authoridx[author2]])) \
            if cond else -1 * gama * global_beta * ((1.0 if author2 in global_author_affils and affil in global_author_affils[author2] else 0.0) - (1.0 if author1 in global_author_affils and affil in global_author_affils[author1] else 0.0)) / ( 1.0 + np.exp(-global_beta * \
                Delta[global_authoridx[author1]][global_authoridx[author2]]))

    delta_obj = np.array([sum([delta_prob((author1, author2, affil, author1 in global_author_authors and author2 in global_author_authors[author1])) \
            for author1 in global_authors for author2 in global_authors if not author1 == author2]) for affil in global_affils])

    return sign * delta_obj


def rank_nodes_mle(authors, author_authors, affils, author_affils, beta=1.0):
    # delta = get_delta(author_affils, author1, author2, affil_score)
    # p = 1.0/( 1.0 + np.exp(-beta * delta))
    global global_authoridx, global_affilidx, global_beta, global_authors, global_affils, global_author_authors, global_author_affils

    global_beta = beta
    global_authors = authors
    global_affils = affils
    global_author_authors = author_authors
    global_author_affils = author_affils

    idx = 0
    author_idx = defaultdict()
    for author in authors:
        author_idx[author] = idx
        idx += 1
    global_authoridx = author_idx


    idx = 0
    affil_idx = defaultdict()
    for affil in affils:
        affil_idx[affil] = idx
        idx += 1
    global_affilidx = affil_idx

    # import pdb;pdb.set_trace()
    # scores, f, d = optimize.fmin_l_bfgs_b(object_func, np.zeros(len(affil_idx)),\
    #              fprime=fprime, bounds=[(0.0, None) for i in xrange(len(affil_idx))], epsilon=1e-04, disp=1)

    # cons = ({'type': 'eq',
    #         'fun' : lambda x: np.array([sum(x) - 1.0]), # x1 + ... + xn = 1
    #         'jac' : lambda x: np.array([1.0 for i in xrange(len(x))])}
    #         )
    cons = ()

    # method = 'SLSQP'
    method = 'L-BFGS-B'

    res = optimize.minimize(object_func, np.zeros(len(affil_idx)), args=(-1.0,), \
            jac=fprime, bounds=[(0.0, 1.0) for i in xrange(len(affil_idx))], \
            constraints=cons, method=method, options={
            'disp': True,
            'factr': 1e7,
            'ftol': 1e-09,
            'maxiter': 50,
            'gtol': 1e-06
            })

    print res
    scores = res.x

    affil_scores = {x:scores[global_affilidx[x]] for x in affils}

    return affil_scores



def rank_nodes_stat(author_graph, author_affils, author_per_paper_dist, author_scores):
    n_paper = 200.0
    new_author_per_paper_dist = {k:int(round(v*n_paper)) for k, v in author_per_paper_dist.iteritems()}
    pred_authorships = []
    author_idx = match_authorid_idx(author_graph)

    W = nx.stochastic_graph(author_graph, weight='weight')

    # shifted distribution
    # epsilon = 1e-6
    # dist = np.array(author_scores.values())
    # new_dist = (dist + epsilon)/sum(dist + epsilon)
    # author_scores = dict(zip(author_scores.keys(), new_dist.tolist()))

    # import pdb;pdb.set_trace()
    for nauthor, npaper in new_author_per_paper_dist.iteritems():
        # 3)
        # import pdb;pdb.set_trace()
        # dependency_coauth_struct = find_dependency_coauth_struct(nauthor, author_scores, W, author_idx)

        for i in xrange(npaper):
            # 1) randomly choose authors
            coauthors = ezpred_coauthors(nauthor, author_scores)

            # 2) randomly choose root author, then search coauthors
            # coauthors = []
            # while not coauthors:
            #     coauthors = pred_coauthors(nauthor, author_scores, W, author_idx)


            # 3) randomly choose dependency coauthor structures
            # dependency_coauth_struct = find_dependency_coauth_struct(nauthor, author_scores, W, author_idx)

            # coauthors = pred_coauth_struct(dependency_coauth_struct)




            pred_authorships.append(coauthors)

    # print 'diff: %s' % len(pred_authorships) - n_paper
    # print len(pred_authorships)

    # affil_scores = calc_affil_scores(pred_authorships, author_affils)
    affil_scores = calc_affil_occurrences(pred_authorships, author_affils)
    return affil_scores


def match_authorid_idx(author_graph):
    author_idx = defaultdict()
    for each in author_graph.nodes():
        author_idx[author_graph.node[each]['entity_id']] = each

    return author_idx




################################################################################
# for each paper, randomly choose a dependency coauthor structure
################################################################################

def pred_coauth_struct(dependency_coauth_struct):
    # np.random.choice only supports 1-dimensional pool
    # do some mapping
    idxs = range(len(dependency_coauth_struct[0]))
    idx = np.random.choice(idxs, 1, replace=False, p=dependency_coauth_struct[1])[0]
    coauthors = dependency_coauth_struct[0][idx]
    return coauthors


def find_dependency_coauth_struct(nauthor, author_scores, author_graph, author_idx):
    dependency_coauth_struct = []

    for root_author in author_scores.keys():
        existing_authors = [root_author]

        # find other authors based on existing authors
        count = 1
        while count < nauthor:
            selector = defaultdict(float)
            for each_existing_author in existing_authors:

                for nbr, weight in author_graph[author_idx[each_existing_author]].iteritems():
                    nbr_id = author_graph.node[nbr]['entity_id']
                    if nbr_id in existing_authors:
                        continue

                    selector[nbr_id] += weight['weight']

            # select the best fit
            if selector:
                # one = max(selector.keys(), key=lambda k: selector[k]) # fixed
                selector = normalize(selector)
                one = np.random.choice(selector.keys(), 1, replace=False, p=selector.values())[0]

                existing_authors.append(one)
                count += 1

            else:
                # print "fails to fetch enough coauthors. retry ..."
                break

        if count == nauthor:
            dependency_coauth_struct.append(sorted(existing_authors))



    # remove duplicate structs
    dependency_coauth_struct.sort()
    dependency_coauth_struct = list(k for k, _ in itertools.groupby(dependency_coauth_struct))

    # rank dependency coauthor structures
    ranking_denpendency_coauth_struct = []
    for struct in dependency_coauth_struct:
        score = 0.0
        for each_author in struct:
            score += author_scores[each_author]

        ranking_denpendency_coauth_struct.append([struct, score])

    # normalize
    tmp = zip(*ranking_denpendency_coauth_struct)
    norm_score = (np.array(tmp[1])/sum(np.array(tmp[1]))).tolist()
    # norm_ranking_dependency_coauth_struct = zip(*[tmp[0], norm_score])

    return [tmp[0], norm_score]




################################################################################
# for each paper, randomly choose a root author, then search his/her coauthors
################################################################################

def pred_coauthors(nauthor, author_scores, author_graph, author_idx):

    # author_graph = author_graph.copy()

    root_author = np.random.choice(author_scores.keys(), 1, replace=False, p=author_scores.values())[0]
    existing_authors = [root_author]

    # find other authors based on existing authors
    count = 1
    while count < nauthor:
        selector = defaultdict(float)
        for each_existing_author in existing_authors:
            # if not author_idx[each_existing_author] in author_graph:
            #     continue

            for nbr, weight in author_graph[author_idx[each_existing_author]].iteritems():
                nbr_id = author_graph.node[nbr]['entity_id']
                if nbr_id in existing_authors:
                    continue

                selector[nbr_id] += weight['weight']

        # select the best fit
        if selector:
            # one = max(selector.keys(), key=lambda k: selector[k]) # fixed
            selector = normalize(selector)
            one = np.random.choice(selector.keys(), 1, replace=False, p=selector.values())[0]

            existing_authors.append(one)
            count += 1

            # author_graph.remove_node(author_idx[one])
            # stochastic_graph(author_graph, copy=False, weight='weight')

        else:
            # print "fails to fetch enough coauthors. retry ..."
            # return []

            # randomly choose one
            one = np.random.choice(author_scores.keys(), 1, replace=False, p=author_scores.values())[0]
            while one in existing_authors:
                one = np.random.choice(author_scores.keys(), 1, replace=False, p=author_scores.values())[0]

            existing_authors.append(one)
            count += 1

            # author_graph.remove_node(author_idx[one])
            # stochastic_graph(author_graph, copy=False, weight='weight')

    return existing_authors



################################################################################
# for each paper, randomly choose all the authors
################################################################################


def ezpred_coauthors(nauthor, author_scores):
    """
    randomly choose authors for a paper based on author citation distribution
    """
    pred_coauthors = np.random.choice(author_scores.keys(), nauthor, replace=False, p=author_scores.values())
    return pred_coauthors





def calc_affil_scores(authorships, author_affils):
    """
    calc affil scores based on authorships
    """
    affil_scores = defaultdict(float)

    for authors in authorships:
        score1 = 1.0 / len(authors)
        for each_author in authors:
            affils = author_affils[each_author]
            if not affils:
                continue

            score2 = score1 / len(affils)
            for each_affil in affils:
                affil_scores[each_affil] += score2

    return affil_scores


def calc_affil_occurrences(authorships, author_affils):
    """
    calc affil scores based on authorships
    """
    affil_occurrences = defaultdict(float)

    # for authors in authorships:
    #     affils = set([y for x in authors for y in author_affils[x]])
    #     for each_affil in affils:
    #         affil_occurrences[each_affil] += 1

    for authors in authorships:
        for each_author in authors:
            affils = author_affils[each_author]
            for each_affil in affils:
                affil_occurrences[each_affil] += 1

    return affil_occurrences


# for IterProjectedLayered approach
# Two-stage projection: Paper -> Author -> Affil
def rank_projected_nodes(graph, alpha=0.3, affil_relev=0.3, out_file=None, stats_file=None, **kwargs):

    # If 'graph' is a string then a path was provided, so we load the graph from it
    if (isinstance(graph, basestring)) :
        graph = nx.read_gexf(graph, node_type=int)


    # Truncate parameters between 0 and 1
    # age_relev = max(min(age_relev, 1.0), 0.0)


    # node_types = {u: layers[graph.node[u]["type"]] for u in graph.nodes()}

#   print graph.number_of_nodes(),

    # Remove isolated nodes
    # graph.remove_nodes_from(nx.isolates(graph))

    print "alpha: %s" % alpha
    print "affil_relev: %s" % affil_relev
    # for u in graph.nodes() :

    #     out_edges = graph.out_edges(u, data=True)

    #     # Here, beside dividing by the total weight for the type of transition, we
    #     # multiply by the probability of the transition, as given by the rho matrix.
    #     for u, v, atts in out_edges:


    # Maps the layers name to the dimensions
    layers = {"affil":0}


    node_types = {u: layers[graph.node[u]["type"]] for u in graph.nodes()}

    # Quick alias method to check if the node is paper
    is_affil = lambda n: (node_types[n]==layers["affil"])


    affil_scores = {}
    naffils = 0
    for u in graph.nodes() :
        if is_affil(u) and graph.node[u].has_key("affil_score"):
            affil_scores[u] = float(graph.node[u]["affil_score"])
            naffils += 1.0


    norm = sum(affil_scores.values())

#   print norm
    if norm == 0.0 :
        uniform_pers = 1.0/graph.number_of_nodes()
        pers = {node: uniform_pers for node in graph.nodes()}

    else:
        uniform_pers = 1.0/naffils
        pers = {}
        for node in graph.nodes() :
            if is_affil(node) :
                pers[node] = affil_relev*(affil_scores[node]/norm) + (1.0-affil_relev)*uniform_pers
            else:
                pers[node] = 0.0

    # Run page rank on the constructed graph
    scores, _niters = pagerank(graph, alpha=(1.0-alpha), pers=pers, node_types=None, max_iter=10000)


    return scores




# for ProjectedLayer approach
def rank_single_layer_nodes(graph, alpha=0.3, out_file=None, stats_file=None, **kwargs):

    # If 'graph' is a string then a path was provided, so we load the graph from it
    if (isinstance(graph, basestring)) :
        graph = nx.read_gexf(graph, node_type=int)


    # Truncate parameters between 0 and 1
    # age_relev = max(min(age_relev, 1.0), 0.0)


    # node_types = {u: layers[graph.node[u]["type"]] for u in graph.nodes()}

#   print graph.number_of_nodes(),

    # Remove isolated nodes
    # import pdb;pdb.set_trace()
    # graph.remove_nodes_from(nx.isolates(graph))

    print "alpha: %s" % alpha
    # for u in graph.nodes() :

    #     out_edges = graph.out_edges(u, data=True)

    #     # Here, beside dividing by the total weight for the type of transition, we
    #     # multiply by the probability of the transition, as given by the rho matrix.
    #     for u, v, atts in out_edges:

    uniform_pers = 1.0/graph.number_of_nodes()
    pers = {node: uniform_pers for node in graph.nodes()}


    # Run page rank on the constructed graph
    scores, _niters = pagerank(graph, alpha=(1.0-alpha), pers=pers, node_types=None, max_iter=10000)

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



def rank_author_affil_nodes(graph, author_affils_relev=0.2, alpha=0.3, affil_relev=0.3, **kwargs):

    # If 'graph' is a string then a path was provided, so we load the graph from it
    if (isinstance(graph, basestring)) :
        graph = nx.read_gexf(graph, node_type=int)

    rho = np.array([
                [1.0-author_affils_relev,             author_affils_relev],
                [         author_affils_relev,       1-author_affils_relev]])

    print rho
    print alpha
    print "affil_relev: %s" % affil_relev
    layers = {"author":0, "affil":1}

    # Alias vector to map nodes into their types (paper, author, etc.) already
    # as their numeric representation (paper=0, author=1, etc.) as listed above.

    node_types = {u: layers[graph.node[u]["type"]] for u in graph.nodes()}
    # Quick alias method to check if the node is paper or affil
    is_affil = lambda n: (node_types[n] == layers["affil"])
    is_author = lambda n: (node_types[n] == layers["author"])

#   print graph.number_of_nodes(),

    # Remove isolated nodes
    # graph.remove_nodes_from(nx.isolates(graph))

#   print graph.number_of_nodes()


    # Normalize weights within each kind of layer transition, e.g., normalize papers to
    # topic edges separately from papers to papers edges.
    for u in graph.nodes() :

        weights = defaultdict(float)
        out_edges = graph.out_edges(u, data=True)
        for u, v, atts in out_edges:

            weights[node_types[v]] = max(atts['weight'], weights[node_types[v]])

        # Here, beside dividing by the total weight for the type of transition, we
        # multiply by the probability of the transition, as given by the rho matrix.
        for u, v, atts in out_edges:
            from_layer = node_types[u]
            to_layer   = node_types[v]

            atts['weight'] *= rho[from_layer][to_layer]/weights[to_layer]


    affil_scores = {}
    naffils = 0

    author_scores = {}
    nauthors = 0
    for u in graph.nodes() :
        if is_affil(u) and graph.node[u].has_key("affil_score"):
            affil_scores[u] = float(graph.node[u]["affil_score"])
            naffils += 1.0
        elif is_author(u) and graph.node[u].has_key("author_score"):
            author_scores[u] = float(graph.node[u]["author_score"])
            nauthors += 1.0


    norm = sum(affil_scores.values())
    norm2 = sum(author_scores.values())

#   print norm
    if norm == 0.0 :
        uniform_pers = 1.0/graph.number_of_nodes()
        pers = {node: uniform_pers for node in graph.nodes()}

    else:
        uniform_pers = 1.0/naffils
        uniform_pers2 = 1.0/nauthors
        # uniform_pers2 = 1.0/(graph.number_of_nodes() - naffils)
        pers = {}
        for node in graph.nodes():
            if is_affil(node):
                pers[node] = affil_relev*(affil_scores[node]/norm) + (1.0-affil_relev)*uniform_pers

            elif is_author(node):
                pers[node] = affil_relev*(author_scores[node]/norm2) + (1.0-affil_relev)*uniform_pers2

            else:
                pers[node] = 0.0


    # Run page rank on the constructed graph
    scores, _niters = pagerank(graph, alpha=(1.0-alpha), pers=pers, node_types=node_types, max_iter=10000)


    return scores



def rank_paper_author_affil_nodes(graph, papers_relev=0.2,
                                         authors_relev=0.2,
                                         author_affils_relev=0.2,
                                         alpha=0.3, affil_relev=0.3, **kwargs):

    # If 'graph' is a string then a path was provided, so we load the graph from it
    if (isinstance(graph, basestring)) :
        graph = nx.read_gexf(graph, node_type=int)



    rho = np.asarray([papers_relev, authors_relev, author_affils_relev])
    # rho = np.asarray([papers_relev, authors_relev, words_relev, author_affils_relev])
    # rho = np.asarray([papers_relev, authors_relev, words_relev, venues_relev, author_affils_relev])

    # Each row and col sumps up to 1
    rho_papers = rho[0] / (rho[0] + rho[1])
    rho_authors = rho[1] / (rho[0] + rho[1])
    rho_affils = rho[2]

    rho = np.array([[rho_papers,     rho_authors,              0],
                [rho_authors,  1.0-rho_authors-rho_affils,             rho_affils],
                [         0,       rho_affils,                 1.0-rho_affils]])

    print rho
    print alpha
    print "affil_relev: %s" % affil_relev
    layers = {"paper":0, "author":1, "affil":2}

    # Alias vector to map nodes into their types (paper, author, etc.) already
    # as their numeric representation (paper=0, author=1, etc.) as listed above.

    node_types = {u: layers[graph.node[u]["type"]] for u in graph.nodes()}
    # Quick alias method to check if the node is paper or affil
    is_affil = lambda n: (node_types[n] == layers["affil"])
    is_author = lambda n: (node_types[n] == layers["author"])
    is_paper = lambda n: (node_types[n] == layers["paper"])

#   print graph.number_of_nodes(),

    # Remove isolated nodes
    # graph.remove_nodes_from(nx.isolates(graph))

#   print graph.number_of_nodes()


    # Normalize weights within each kind of layer transition, e.g., normalize papers to
    # topic edges separately from papers to papers edges.
    for u in graph.nodes() :

        weights = defaultdict(float)
        out_edges = graph.out_edges(u, data=True)
        for u, v, atts in out_edges:

            weights[node_types[v]] = max(atts['weight'], weights[node_types[v]])

        # Here, beside dividing by the total weight for the type of transition, we
        # multiply by the probability of the transition, as given by the rho matrix.
        for u, v, atts in out_edges:
            from_layer = node_types[u]
            to_layer   = node_types[v]

            atts['weight'] *= rho[from_layer][to_layer]/weights[to_layer]


    affil_scores = {}
    naffils = 0
    for u in graph.nodes() :
        if is_affil(u) and graph.node[u].has_key("affil_score"):
            affil_scores[u] = float(graph.node[u]["affil_score"])
            naffils += 1.0


    norm = sum(affil_scores.values())

#   print norm
    if norm == 0.0 :
        uniform_pers = 1.0/graph.number_of_nodes()
        pers = {node: uniform_pers for node in graph.nodes()}

    else:
        uniform_pers = 1.0/naffils
        # uniform_pers2 = 1.0/(graph.number_of_nodes() - naffils)
        pers = {}
        for node in graph.nodes() :
            if is_affil(node) :
                pers[node] = affil_relev*(affil_scores[node]/norm) + (1.0-affil_relev)*uniform_pers
            else:
                pers[node] = 0.0


    # Run page rank on the constructed graph
    scores, _niters = pagerank(graph, alpha=(1.0-alpha), pers=pers, node_types=node_types, max_iter=10000)


    return scores



# for MultiLayered approach
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
                [         0,       1,                 0]])
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
    current_year = PARAMS['current_year']
    old_year = PARAMS['old_year']
    years = []
    for u in graph.nodes() :
        if is_paper(u) and graph.node[u].has_key("year"):
            if (graph.node[u]["year"] > 0) :
                years.append(graph.node[u]["year"])

    year_median = np.median(years)

    # log.debug("Using year=%d (median) for missing values." % int(year_median))


    # Normalize weights within each kind of layer transition, e.g., normalize papers to
    # topic edges separately from papers to papers edges.
    print "age_relev: %s"%age_relev
    for u in graph.nodes() :

        weights = defaultdict(float)
        out_edges = graph.out_edges(u, data=True)
        for u, v, atts in out_edges:

        # Also apply the age attenuator to control relevance of old and highly cited papers
            if is_paper(v) and is_paper(u):
                # 1)
                if graph.node[v].has_key("year"):
                    year = graph.node[v]["year"] # year of u or v?

                    if year == 0:
                        year = year_median
                    else:
                        year = min(max(year, old_year), current_year)
                else:
                    year = year_median

                # weight = 1.0
                weight = np.exp(-(age_relev)*(current_year-year))



                # # 2)
                # yu = graph.node[u]["year"] if graph.node[u].has_key("year") else year_median
                # yv = graph.node[v]["year"] if graph.node[v].has_key("year") else year_median
                # diff = abs(yu - yv)

                # weight = np.exp(-(age_relev)*diff)



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

def normalize(x):
  sums = sum(x.values())
  xx = {k:v/sums for k, v in x.iteritems()}
  return xx



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

