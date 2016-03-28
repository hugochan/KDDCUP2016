'''
Created on Mar 27, 2016

@author: hugo
'''

from mymysql.mymysql import MyMySQL
import config
import re

db = MyMySQL(config.DB_NAME, user=config.DB_USER, passwd=config.DB_PASSWD)

def clean_dblp_author_affils(table_name):
    rst = db.select_query("select count(*) from %s"%table_name)[0][0]
    bulk_size = 10000
    nbulk = rst/bulk_size
    count = 0

    for idx in range(nbulk):
        rows = db.select_query("select * from %s limit %s offset %s"%(table_name, bulk_size, idx*bulk_size))
        for each_row in rows: # id | name | other_names | affil_name
            new_name = re.sub(" \d+", " ", each_row[1]).strip()
            if new_name != each_row[1]:
                try:
                    db.update(table_name, set="name='%s'"%new_name, where="id=%s"%each_row[0])
                    count += 1
                except Exception, e:
                    print e
        print "%s bulks processed."%(idx+1)

    if nbulk * bulk_size < rst:
        rows = db.select_query("select * from %s limit %s offset %s"%(table_name, bulk_size, nbulk*bulk_size))
        for each_row in rows:
            new_name = re.sub(" \d+", " ", each_row[1]).strip()
            if new_name != each_row[1]:
                try:
                    db.update(table_name, set="name='%s'"%new_name, where="id=%s"%each_row[0])
                    count += 1
                except Exception, e:
                    print e
    print "%s rows affected." % count



def clean_csx_author_affils(table_name):
    rst = db.select_query("select count(*) from %s"%table_name)[0][0]
    bulk_size = 100000
    nbulk = rst/bulk_size
    count = 0

    for idx in range(nbulk):
        rows = db.select_query("select * from %s limit %s offset %s"%(table_name, bulk_size, idx*bulk_size))
        for each_row in rows: # id | cluster | name | affil | address | email | ord | paperid
            new_name = each_row[2].strip().strip('\t').strip()
            if new_name != each_row[2]:
                try:
                    db.update(table_name, set="name='%s'"%new_name, where="id=%s"%each_row[0])
                    count += 1
                except Exception, e:
                    print e
        print "%s bulks processed."%(idx+1)

    if nbulk * bulk_size < rst:
        rows = db.select_query("select * from %s limit %s offset %s"%(table_name, bulk_size, nbulk*bulk_size))
        for each_row in rows:
            new_name = each_row[2].strip().strip('\t').strip()
            if new_name != each_row[2]:
                try:
                    db.update(table_name, set="name='%s'"%new_name, where="id=%s"%each_row[0])
                    count += 1
                except Exception, e:
                    print e
    print "%s rows affected." % count



if __name__ == '__main__':
    # clean_dblp_author_affils('dblp_author_affils')
    clean_csx_author_affils('csx_authors')
