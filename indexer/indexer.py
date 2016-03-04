'''
Created on Jul 6, 2014

@author: luamct
'''
from mymysql import MyMySQL
import os
from pylucene import Index, DocField
import time
import config
import logging as log


class Indexer:

  def __init__(self):
    self.db = MyMySQL(db=config.DATASET,
                      user=config.DB_USER,
                      passwd=config.DB_PASSWD)

#		self.pubs = {}
#		rows = db.select(["id", "title", "abstract"], table="papers", where="use_it=1")
#		for id, title, abs in rows :
#			self.pubs[str(id)] = ((title if title else ""), (abs if abs else ""))

    self.pub_ids = self.db.select("id", table="papers")
    print "Ids loaded."


  def get_texts(self, pub_id) :
    title, abs = self.db.select_one(["title", "abstract"], table="papers", where="id='%s'"%pub_id)
    title = title if title else ''
    abs = abs if abs else ''
    return title, abs


  def add_papers(self, index_folder, include_text=True):

    print "Adding %s documents to index in '%s'" % (len(self.pub_ids), index_folder)

    fields = [DocField("id", stored=True, indexed=True),
              DocField("title", stored=True, indexed=True),
              DocField("abstract", stored=False, indexed=True)]
    if include_text:
      fields.append(DocField("text", stored=False, indexed=True))

    index = Index(index_folder, fields)
#		for i, (id, (title, abstract)) in enumerate(self.pubs.items()) :
    for i, pub_id in enumerate(self.pub_ids) :

      title, abstract = self.get_texts(pub_id)
      field_values = {'id':pub_id, 'title':title, 'abstract':abstract}

      # Check if we are including to text before loading it
      if include_text :
        with open(os.path.join(config.TXT_PATH % pub_id), "r") as txt_file :
          text = txt_file.read()
        field_values['text'] = text

      index.add(**field_values)

      # Commit and print progress every 1000 entries
      if i and i%1000==0 :
        index.commit()
        log.info("%d documents added." % i)

    index.commit()
    index.close()


def search_index(index_folder, query) :
  index = Index(index_folder)
  top = index.search("text", query, fields=["id", "title"])
  print "\n".join(map(str,top))


if __name__ == '__main__':

  start = time.time()

  indexer = Indexer()
  indexer.add_papers(config.INDEX_PATH, include_text=False)

  print "Process finished in %.2f seconds." % (time.time()-start)

# 	search_index(config.DATA + "index_csx", "data clustering")

