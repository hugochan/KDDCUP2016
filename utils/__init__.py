'''
Created on Apr 20, 2014

@author: luamct
'''
import sys
from mymysql.mymysql import MyMySQL
import config
from collections import defaultdict 
import nltk
import re
import os
import random


stemmer = nltk.stem.porter.PorterStemmer()
stopwords = set(nltk.corpus.stopwords.words('english'))

# This regex only matches 2 or more characters tokens
TOKEN_REGEX  = re.compile(r"(?u)\b\w\w+\b")
NUMBER_REGEX = re.compile("\d+$")


class PubTexts():

  def __init__(self, n=None):
    db = MyMySQL(db=config.DB_NAME,
                 user=config.DB_USER,
                 passwd=config.DB_PASSWD)

    rows = db.select(fields=["id", "title", "abstract"], table="papers")
    if n :
      rows = random.sample(rows, n)

    self.pubs = {str(id): (title, abs) for id, title, abs in rows}


  def ids(self):
    return self.pubs.keys()


  def texts(self, pub_ids=None, use_title=True, use_abs=True, use_body=False) :

    # If list of ids is not given, use all of them
    if (pub_ids is None):
      pub_ids = self.pubs.keys()

    texts = []
    for pub_id in pub_ids :

      text = []
      if use_title or use_abs:

        title, abs = self.pubs[str(pub_id)]
        if use_title and title: text.append(title)
        if use_abs and abs : text.append(abs)

      if use_body :
        pass  # TODO

      texts.append(" ".join(text))
    return texts


def get_texts(pub_ids, use_title=True, use_abs=True) :
  '''
  This is a non-batch version. Much slower but more
  memory efficient.
  '''
  db = MyMySQL(db='csx', user='root', passwd='')

  fields = []
  if use_title: fields.append("title")
  if use_abs: fields.append("abstract")

  texts = []
  for pub_id in pub_ids:
    text_fields = db.select_one(fields=fields, table="papers", where="id='%s'" % pub_id)
    text = ''
    for tf in text_fields:
      if tf is not None:
        text += tf

    texts.append(text)

  return texts


def tokenize(text) :
  """
  Tokenizes input string using TOKEN_REGEX expression. Sets to lower string, removes
  english stop words, ignore numerical tokens and stems the output.
  """
  tokens = []
  for token in TOKEN_REGEX.findall(text.lower()) :

    # If not a stop word or a number, apply stemmer and append.
    if (token not in stopwords) and \
       (not re.match(NUMBER_REGEX, token)) :
      tokens.append(stemmer.stem(token))

  return tokens


def progress(items, step=1000) :
  '''
  Simple decorator to print progress every <step> iterations.
  '''
  for i, item in enumerate(items) :
    if (i%step==0) and i :
      print "%d itens processed" % i
      sys.stdout.flush()
    yield item


def read_text(doc_id):
  '''
  Simple loads text from given document.
  '''
  with open(config.TXT_PATH % doc_id, 'r') as f :
    return unicode(f.read(), "UTF-8")


def get_graph_file_name(query, folder=None, extension="gexf") :
  '''
  Strip empty spaces from extremities, convert to lower case
  and replace problematic characters for the file system by +
  '''
  query = query.strip().lower()
  path = query.replace(" ", "+").replace("/", "+")
  if extension:
    path = "%s.%s" % (path, extension)

  if folder :
    path = os.path.join(folder, path)

  return path



def get_title(db, pub_id):
  return db.select_one("title", table="papers", where="id='%s'"%pub_id)


def get_cited(db, pub_id) :
  return db.select("cited", table="graph", where="citing='%s'"%pub_id)


def config_logging(name=None, file_path=None, stream=None, level=None, format=None, datefmt=None) :
  import logging as log

  handlers = []
  logger = log.getLogger(name)
  if (stream != None):
    handlers.append(log.StreamHandler(stream))

  if (file_path != None) :
    handlers.append(log.FileHandler(file_path))

  if (level != None) :
    logger.setLevel(level)

  for handler in handlers:

    if (format!=None) :
      handler.setFormatter(log.Formatter(fmt=format, datefmt=datefmt))

    logger.addHandler(handler)

  return logger


def ensure_folder(folder):
  ''' Creates given folder if non-existing '''
  if not os.path.exists(folder) :
    os.makedirs(folder)


