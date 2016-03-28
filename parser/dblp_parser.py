'''
Created on Mar 26, 2016

@author: hugo
A DOM-based parser
'''

import xml.dom.minidom
from mymysql.mymysql import MyMySQL
import config
import sys

db = MyMySQL(config.DB_NAME, user=config.DB_USER, passwd=config.DB_PASSWD)


def parse(in_file, table_name="dblp_author_affils"):
   # create table
   table_description = ['id INT NOT NULL AUTO_INCREMENT',
                        'name VARCHAR(200) NOT NULL',
                        'other_names VARCHAR(1000)',
                        'affil_name VARCHAR(200)',
                        'PRIMARY KEY (id)',
                        'KEY (name)']

   db.create_table(table_name, table_description)

   fields = ["name", "other_names", "affil_name"]

   # Open XML document using minidom parser
   DOMTree = xml.dom.minidom.parse(in_file)
   dblp = DOMTree.documentElement

   # Get all the movies in the collection
   wwws = dblp.getElementsByTagName("www")

   nauthor_affils = 0
   author_affils = []
   for www in wwws:
      if www.hasAttribute("key"):
         key = www.getAttribute("key")

         if not "homepages" in key.split('/'):
            continue

         try:
            notes = www.getElementsByTagName('note')
            for note in notes:
               if note.hasAttribute("type") and \
                  note.getAttribute("type") == "affiliation":
                  affil_name = note.childNodes[0].data
                  # print "affil: %s" % affil_name

                  authors = www.getElementsByTagName('author')
                  if authors:
                     author_names = [author.childNodes[0].data for author in authors]
                     # print "Authors: %s" % author_names
                     author_affils.append((author_names[0], '/'.join([str(x) for x in author_names[1:]]), affil_name))
                     nauthor_affils += 1
                  break
         except Exception, e:
            pass

      # write to db
      if len(author_affils) == 10000:
         db.insert(into=table_name, fields=fields, values=author_affils, ignore=True)
         author_affils[:] = []   # Empty the list
         print "%d processed." % nauthor_affils

   db.insert(into=table_name, fields=fields, values=author_affils, ignore=True)
   print "%d processed." % nauthor_affils


# def parse2(path_to_file):
#    for event, element in etree.iterparse(path_to_file, tag="www"):
#     for child in element:
#         print child.tag, child.text
#     element.clear()

if __name__ == '__main__':
   try:
      in_file = sys.argv[1]
   except Exception as e:
      print e
      sys.exit()

   # parse(in_file)
   parse2(in_file)

