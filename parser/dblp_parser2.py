'''
Created on Mar 27, 2016

@author: hugo
An event-driven parser
'''

import xml.sax
from mymysql.mymysql import MyMySQL
import config
import subprocess
import sys
import re

total_lineno = None

db = MyMySQL(config.DB_NAME, user=config.DB_USER, passwd=config.DB_PASSWD)
auth_affil_bulk = set()

class DBLPHandler(xml.sax.ContentHandler):
    def __init__(self, locator, table_name, fields):
        self.loc = locator
        self.setDocumentLocator(self.loc)
        self.CurrentData = ""
        self.valid = False # check if www tag
        self.is_affil = False # check if valid (having affiliation attr) note tag
        self.authors = set()
        self.affils = set()
        # self.auth_affil_bulk = set()
        self.valid_count = 0 # num of homepage records
        self.author_count = 0 # num of valid (i.e., having affils) authors
        self.count = 0 # num of tags


    # # Weird errors, does not work, so we adopt a stupid way here, set auth_affil_bulk as global
    # def __del__(self):
    #     super(xml.sax.ContentHandler, self).__del__()
    #     if self.auth_affil_bulk:
    #         db.insert(into=self.table_name, fields=self.fields, values=self.auth_affil_bulk, ignore=True)


    # Call when an element starts
    def startElement(self, tag, attr):
        self.count += 1
        if self.count % 1000000 == 0:
            print "progress: %.2f %%" % (self.get_progress()*100)

        self.CurrentTag = tag
        if tag == "www" and attr.has_key("key") \
                and "homepages" in attr["key"].split("/"):
            # www homepage tage
            self.valid = True
            self.valid_count += 1

        elif self.valid and tag == "note" and attr.has_key("type") \
                and attr["type"] == "affiliation":
            # affiliation
            self.is_affil = True


    # Call when an elements ends
    def endElement(self, tag):
        if self.valid and tag == "www":
            self.valid = False # reset flag
            # pack data
            if self.affils:
                affil_names = self.affils

                if self.authors:
                    author_name = list(self.authors)[0]
                    author_name = re.sub(" \d+", " ", author_name) # remove digits at the end of the string
                    other_names = list(self.authors)[1:]
                    # write to db
                    # print "author name: %s" % author_name
                    # print "other names: %s" % other_names
                    # print "affil names: %s" % affil_name
                    # print

                    # one author may have multiple affils, we store all of them
                    auth_affils = set()
                    for each in affil_names:
                        auth_affils.add((author_name, '/'.join(x for x in other_names]), each))

                    auth_affil_bulk.update(auth_affils)

                    if len(auth_affil_bulk) % 500 == 0:
                        db.insert(into=table_name, fields=fields, values=list(auth_affil_bulk), ignore=True)
                        auth_affil_bulk.clear()

                    self.author_count += 1
                    if self.author_count % 1000 == 0:
                        print "%s valid (having affils) authors processed." % self.author_count


            self.authors.clear()
            self.affils.clear()

            if self.valid_count % 1000 == 0:
                print "%s homepages processed."%self.valid_count


        elif self.is_affil and tag == "note":
            self.is_affil = False

        # elif self.CurrentTag == "author":
            # print "Author:", self.author
        # elif self.CurrentTag == "note":
            # print "Note:", self.note
        # elif self.CurrentTag == "title":
            # print "Title:", self.title
        # elif self.CurrentTag == "url":
            # print "URL:", self.url

        self.CurrentTag = ""

    # Call when a character is read
    def characters(self, content):
        if self.valid and self.CurrentTag == "author":
            self.authors.add(content.strip('\r\n').strip())
        if self.is_affil and self.CurrentTag == "note":
            self.affils.add(content.strip('\r\n').strip())
        # elif self.CurrentTag == "title":
            # self.title = content
        # elif self.CurrentTag == "url":
            # self.url = content

    def get_progress(self):
        global total_lineno
        return self.loc.getLineNumber()/float(total_lineno)


# get total line num of a file
def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE,
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])


if __name__ == "__main__":
    try:
        in_file = sys.argv[1]
    except Exception, e:
        print e
        sys.exit()

    total_lineno = file_len(in_file)
    print "total line num of the file: %s\n" % total_lineno

    # db info
    table_description = ['id INT NOT NULL AUTO_INCREMENT',
                        'name VARCHAR(200) NOT NULL',
                        'other_names VARCHAR(1000)',
                        'affil_name VARCHAR(200)',
                        'PRIMARY KEY (id)',
                        'KEY (name)']

    table_name = "dblp_author_affils"
    fields = ["name", "other_names", "affil_name"]

    # create table
    db.create_table(table_name, table_description, force=True)


    # create an XMLReader
    parser = xml.sax.make_parser()
    locator = xml.sax.expatreader.ExpatLocator(parser)
    # turn off namepsaces
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    # override the default ContextHandler
    Handler = DBLPHandler(locator, table_name, fields)
    parser.setContentHandler(Handler)
    parser.parse(in_file)

    # write remaining data into db
    if auth_affil_bulk:
        db.insert(into=table_name, fields=fields, values=list(auth_affil_bulk), ignore=True)

    print "It's done."
