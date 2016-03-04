from mymysql.mymysql import MyMySQL
from utils import progress


def get_pubs(db):
  return db.select(["id", "cluster"], table="papers")


def get_citations(db):
  return db.select(["citing", "cited"], table="graph", where="context is NULL")


def get_context(db, cciting, ccited):
  return db.select_one("firstContext", table="citegraph",
                       where="(citing=%d) AND (cited=%d)" % (cciting, ccited))


def update_graph(db, citing, cited, context):
  db.update(table="graph",
            set="context='%s'" % context,
            where="(citing='%s') AND (cited='%s')" % (citing, cited))


def fix_contexts_limits() :
  """
  Updates the contexts on the graph table so that the tokens on the
  extremities are removed. These are usually parts of words, and therefore
  are meaningless.
  """
  db = MyMySQL(db="csx", user="root", passwd="")
  ctxs = db.select(["citing", "cited", "context"], table="graph", where="context != ''")

  print len(ctxs)
  for citing, cited, ctx in progress(ctxs):
    s = ctx.find(" ")
    e = ctx.rfind(" ")

    # print ctx
    # print ctx[s+1:e]
    # print

    db.update(table="graph",
              set="context='%s'" % ctx[s+1:e],
              where="(citing='%s') AND (cited='%s')" % (citing, cited))


def load_contexts():

  db_csx = MyMySQL(db="csx", user="root", passwd="")
  db_cg = MyMySQL(db="csx_citegraph", user="root", passwd="")

  pubs = get_pubs(db_csx)
  print "%d publications loaded." % len(pubs)

  clusters = {str(pub_id): cluster for pub_id, cluster in pubs}

  citations = get_citations(db_csx)
  print "%d citations loaded." % len(citations)

  found = 0
  for n, (citing, cited) in enumerate(citations):
    cciting = clusters[str(citing)]
    ccited  = clusters[str(cited)]
    context = get_context(db_cg, cciting, ccited)

    if context is None:
      context = ''
    else:
      context = context.replace("'", '"')
      found += 1

    # if (context is not None) and (context != ""):
    try:
      update_graph(db_csx, citing, cited, context)
    except:
      print "Exception when updating 'graph' table."


    print "%d out of %d contexts found." % (found, n + 1)


# if __name__ == '__main__':
#   fix_contexts_limits()
