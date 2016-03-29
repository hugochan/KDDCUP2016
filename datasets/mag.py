'''
Created on Mar. 10, 2016

@author: hugo
'''
from collections import defaultdict
from mymysql.mymysql import MyMySQL
from exceptions import TypeError
import sys
import re
import config
import chardet
from datasets.affil_names import *



db = MyMySQL(config.DB_NAME, user=config.DB_USER, passwd=config.DB_PASSWD)


# regexp patterns
univ_academy_pattern = '([A-Z][^\\s,.:;]+[.]?\\s[(]?)*(University|Universidade|Academy|Council)[^,;:.\\d]*(?=,|\\d|;|-|\\.)'
institute_pattern = '([A-Z][^\\s,.:;]+[.]?\\s[(]?)*(Institute)[^,;:.\\d]*(?=,|\\d|;|-|\\.)'
college_pattern = '([A-Z][^\\s,.:;]+[.]?\\s[(]?)*(College|Centre|Center)[^,;:.\\d]*(?=,|\\d|;|-|\\.)'

univ_academy_pattern2 = '([A-Z][^\\s,.:;]+[.]?\\s[(]?)*(University|Universidade|Academy|Council)[^,;:.\\d]*$'
institute_pattern2 = '([A-Z][^\\s,.:;]+[.]?\\s[(]?)*(Institute)[^,;:.\\d]*$'
college_pattern2 = '([A-Z][^\\s,.:;]+[.]?\\s[(]?)*(College|Centre|Center)[^,;:.\\d]*$'


univ_academy_prog = re.compile(univ_academy_pattern)
institute_prog = re.compile(institute_pattern)
college_prog = re.compile(college_pattern)

univ_academy_prog2 = re.compile(univ_academy_pattern2)
institute_prog2 = re.compile(institute_pattern2)
college_prog2 = re.compile(college_pattern2)



def import_selected_affils(file_path, table_name='selected_affils'):
    table_description = ['id VARCHAR(30) NOT NULL',
                        'name VARCHAR(200) NOT NULL',
                        'PRIMARY KEY (id)',
                        'UNIQUE (name)']
    db.create_table(table_name, table_description)

    naffils = 0
    affils = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip('\r\n').strip(' ')
                if line == '':
                    continue

                # Affiliation ID
                # Affiliation name
                try :
                    id, name = line.split('\t')
                    affils.append((id, name))
                    naffils += 1

                    # Buffer to avoid DB accesses
                    if len(affils) == 100:
                        db.insert(into=table_name, fields=["id", "name"], values=affils, ignore=True)
                        affils[:] = []   # Empty the list
                        print "%d processed." % naffils


                # If an exception occurs, there was no data to be
                # processed, so just skip it
                except:
                    pass

            db.insert(into=table_name, fields=["id", "name"], values=affils, ignore=True)
            print "totally %d processed." % naffils

    except Exception, e:
        print e
        sys.exit()

    f.close()

def import_selected_papers(file_path, table_name='selected_papers'):
    table_description = ['id VARCHAR(30) NOT NULL',
                        'title VARCHAR(300) NOT NULL',
                        'year SMALLINT',
                        'venue_id VARCHAR(30)',
                        'venue_abbr_name VARCHAR(30)',
                        'PRIMARY KEY (id)',
                        'KEY (title)',
                        'KEY (year)',
                        'KEY (venue_id)']
    db.create_table(table_name, table_description)

    npapers = 0
    papers = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip('\r\n').strip(' ')
                if line == '':
                    continue

                # Paper ID
                # Original paper title
                # Paper publish year
                # Conference series ID mapped to venue name
                # Conference series short name (abbreviation)
                try :
                    id, title, year, venue_id, venue_abbr_name = line.split('\t')
                    # encoding = chardet.detect(title)['encoding']
                    # title = unicode(title, encoding).encode('utf-8')
                    papers.append((id, title, year, venue_id, venue_abbr_name))
                    npapers += 1

                    # # convert to utf-8, get mysql warnings when storing special char
                    # # if npapers == 2118:
                    # #     import pdb;pdb.set_trace()
                    # #     import chardet;
                    # #     coding=chardet.detect(title)['encoding']
                    # #     title=unicode(title,coding).encode('utf-8')
                    # db.insert(into=table_name, fields=["id", "title", "year", "venue_id", "venue_abbr_name"], values=[(id, title, year, venue_id, venue_abbr_name)], ignore=True)
                    # print "%d processed." % npapers

                    # Buffer to avoid DB accesses
                    if len(papers) == 100:
                        db.insert(into=table_name, fields=["id", "title", "year", "venue_id", "venue_abbr_name"], values=papers, ignore=True)
                        papers[:] = []   # Empty the list
                        print "%d processed." % npapers


                # If an exception occurs, there was no data to be
                # processed, so just skip it
                except:
                    pass

            db.insert(into=table_name, fields=["id", "title", "year", "venue_id", "venue_abbr_name"], values=papers, ignore=True)
            print "totally %d processed." % npapers

    except Exception, e:
        print e
        sys.exit()

    f.close()


def import_affils(file_path, table_name='affils'):
    table_description = ['id VARCHAR(30) NOT NULL',
                        'name VARCHAR(200) NOT NULL',
                        # 'UNIQUE (name)', # there are duplicate names in this dataset, so turn-off unique key here
                        'PRIMARY KEY (id)']
    db.create_table(table_name, table_description)

    naffils = 0
    affils = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip('\r\n').strip(' ')
                if line == '':
                    continue

                # Affiliation ID
                # Affiliation name
                try :
                    id, name = line.split('\t')
                    affils.append((id, name))
                    naffils += 1

                    if len(affils) == 1000:
                        db.insert(into=table_name, fields=["id", "name"], values=affils, ignore=True)
                        affils[:] = []   # Empty the list
                        print "%d processed." % naffils


                # If an exception occurs, there was no data to be
                # processed, so just skip it
                except:
                    pass

            db.insert(into=table_name, fields=["id", "name"], values=affils, ignore=True)
            print "totally %d processed." % naffils

    except Exception, e:
        print e
        sys.exit()


def import_confs(file_path, table_name='confs'):
    table_description = ['id VARCHAR(30) NOT NULL',
                        'abbr_name VARCHAR(30) NOT NULL',
                        'full_name VARCHAR(200) NOT NULL',
                        'PRIMARY KEY (id)']
    db.create_table(table_name, table_description)

    nconfs = 0
    confs = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip('\r\n').strip(' ')
                if line == '':
                    continue

                # Conference series ID
                # Short name (abbreviation)
                # Full name
                try :
                    id, abbr_name, full_name = line.split('\t')
                    confs.append((id, abbr_name, full_name))
                    nconfs += 1

                    if len(confs) == 1000:
                        db.insert(into=table_name, fields=["id", "abbr_name", "full_name"], values=confs, ignore=True)
                        confs[:] = []   # Empty the list
                        print "%d processed." % nconfs


                # If an exception occurs, there was no data to be
                # processed, so just skip it
                except:
                    pass

            db.insert(into=table_name, fields=["id", "abbr_name", "full_name"], values=confs, ignore=True)
            print "totally %d processed." % nconfs

    except Exception, e:
        print e
        sys.exit()

    f.close()


def import_conf_insts(file_path, table_name='conf_insts'):
    table_description = ['conf_id VARCHAR(30)',
                        'id VARCHAR(30) NOT NULL',
                        'abbr_name VARCHAR(100) NOT NULL',
                        'full_name VARCHAR(300) NOT NULL',
                        'location VARCHAR(200)',
                        'url VARCHAR(200)',
                        'start_date VARCHAR(20)',
                        'end_date VARCHAR(20)',
                        'abstract_registr_date VARCHAR(20)',
                        'submit_deadline_date VARCHAR(20)',
                        'notify_due_date VARCHAR(20)',
                        'final_due_date VARCHAR(20)',
                        'PRIMARY KEY (id)',
                        'KEY (conf_id)']
    db.create_table(table_name, table_description)

    fields = ["conf_id", "id", "abbr_name", "full_name",
                "location", "url", "start_date", "end_date", "abstract_registr_date",
                "submit_deadline_date", "notify_due_date", "final_due_date"]
    nconfinsts = 0
    confinsts = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip('\r\n').strip(' ') # there are missing values in this dataset, keep \t sign for locating
                if line == '':
                    continue

                # There are duplicate records in this dataset
                # Conference series ID
                # Short name (abbreviation)
                # Full name
                try :
                    row = line.split('\t')
                    confinsts.append(tuple(row))
                    nconfinsts += 1

                    if len(confinsts) == 1000:
                        # db.insert(into=table_name, fields=fields, values=confinsts, ignore=True)
                        confinsts[:] = []   # Empty the list
                        print "%d processed." % nconfinsts

                # If an exception occurs, there was no data to be
                # processed, so just skip it
                except:
                    pass

            db.insert(into=table_name, fields=fields, values=confinsts, ignore=True)
            print "totally %d processed." % nconfinsts

    except Exception, e:
        print e
        sys.exit()

    f.close()


def import_journals(file_path, table_name='journals'):
    table_description = ['id VARCHAR(30) NOT NULL',
                        'name VARCHAR(200) NOT NULL',
                        'PRIMARY KEY (id)']
    db.create_table(table_name, table_description)

    njournals = 0
    journals = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip('\r\n').strip(' ')
                if line == '':
                    continue

                # Journal ID
                # Journal name
                try :
                    id, name = line.split('\t')
                    journals.append((id, name))
                    njournals += 1

                    if len(journals) == 1000:
                        db.insert(into=table_name, fields=["id", "name"], values=journals, ignore=True)
                        journals[:] = []   # Empty the list
                        print "%d processed." % njournals


                # If an exception occurs, there was no data to be
                # processed, so just skip it
                except:
                    pass

            db.insert(into=table_name, fields=["id", "name"], values=journals, ignore=True)
            print "totally %d processed." % njournals

    except Exception, e:
        print e
        sys.exit()

    f.close()


def import_authors(file_path, table_name='authors'):
    table_description = ['id VARCHAR(30) NOT NULL',
                        'name VARCHAR(200) NOT NULL',
                        'PRIMARY KEY (id)']
    db.create_table(table_name, table_description)

    nauthors = 0
    authors = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip('\r\n').strip(' ')
                if line == '':
                    continue

                # Author ID
                # Author name
                try :
                    id, name = line.split('\t')
                    authors.append((id, name))
                    nauthors += 1

                    if len(authors) == 100000:
                        db.insert(into=table_name, fields=["id", "name"], values=authors, ignore=True)
                        authors[:] = []   # Empty the list
                        print "%d processed." % nauthors


                # If an exception occurs, there was no data to be
                # processed, so just skip it
                except:
                    pass

            db.insert(into=table_name, fields=["id", "name"], values=authors, ignore=True)
            print "totally %d processed." % nauthors

    except Exception, e:
        print e
        sys.exit()

    f.close()


def import_paper_keywords(file_path, table_name='paper_keywords'):
    table_description = ['id INT NOT NULL AUTO_INCREMENT',
                        'paper_id VARCHAR(30) NOT NULL',
                        'keyword_name VARCHAR(200)',
                        'field_of_study_id VARCHAR(30)',
                        'PRIMARY KEY (id)',
                        'KEY (paper_id)',
                        'KEY (field_of_study_id)']
    db.create_table(table_name, table_description)

    nkeywords = 0
    keywords = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip('\r\n').strip(' ')
                if line == '':
                    continue

                # Paper ID
                # Keyword name
                # Field of study ID mapped to keyword
                try :
                    paper_id, keyword_name, field_of_study_id = line.split('\t')
                    keywords.append((paper_id, keyword_name, field_of_study_id))
                    nkeywords += 1

                    if len(keywords) == 100000:
                        db.insert(into=table_name, fields=["paper_id", "keyword_name", "field_of_study_id"], values=keywords, ignore=True)
                        keywords[:] = []   # Empty the list
                        print "%d processed." % nkeywords


                # If an exception occurs, there was no data to be
                # processed, so just skip it
                except:
                    pass

            db.insert(into=table_name, fields=["paper_id", "keyword_name", "field_of_study_id"], values=keywords, ignore=True)
            print "totally %d processed." % nkeywords

    except Exception, e:
        print e
        sys.exit()

    f.close()


def import_paper_refs(file_path, table_name='paper_refs'):
    table_description = ['id INT NOT NULL AUTO_INCREMENT',
                        'paper_id VARCHAR(30) NOT NULL',
                        'paper_ref_id VARCHAR(30) NOT NULL',
                        'PRIMARY KEY (id)',
                        'KEY (paper_id)',
                        'KEY (paper_ref_id)']
    db.create_table(table_name, table_description)

    npaper_refs = 0
    paper_refs = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip('\r\n').strip(' ')
                if line == '':
                    continue

                # Paper ID
                # Paper reference ID
                try :
                    paper_id, paper_ref_id = line.split('\t')
                    paper_refs.append((paper_id, paper_ref_id))
                    npaper_refs += 1

                    if len(paper_refs) == 100000:
                        db.insert(into=table_name, fields=["paper_id", "paper_ref_id"], values=paper_refs, ignore=True)
                        paper_refs[:] = []   # Empty the list
                        print "%d processed." % npaper_refs


                # If an exception occurs, there was no data to be
                # processed, so just skip it
                except:
                    pass

            db.insert(into=table_name, fields=["paper_id", "paper_ref_id"], values=paper_refs, ignore=True)
            print "totally %d processed." % npaper_refs

    except Exception, e:
        print e
        sys.exit()

    f.close()


def import_papers(file_path, table_name='papers'):
    table_description = ['id VARCHAR(30) NOT NULL',
                        'title VARCHAR(300) NOT NULL',
                        'normal_title VARCHAR(300)',
                        'year SMALLINT',
                        'date VARCHAR(20)',
                        'DOI VARCHAR(100)',
                        'venue_name VARCHAR(200)',
                        'normal_venue_name VARCHAR(200)',
                        'jornal_id VARCHAR(30)',
                        'conf_id VARCHAR(30)',
                        'paper_rank MEDIUMINT',
                        'PRIMARY KEY (id)',
                        'UNIQUE (DOI)',
                        'KEY (jornal_id)',
                        'KEY (conf_id)']
    db.create_table(table_name, table_description)
    fields = ['id', 'title', 'normal_title', 'year', 'date', 'DOI', 'venue_name',
            'normal_venue_name', 'jornal_id', 'conf_id', 'paper_rank']
    nfields = len(fields)

    npapers = 0
    papers = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip('\r\n').strip(' ')
                if line == '':
                    continue
                # 126909021 records
                # Paper ID
                # Original paper title
                # Normalized paper title
                # Paper publish year
                # Paper publish date
                # Paper Document Object Identifier (DOI)
                # Original venue name
                # Normalized venue name
                # Journal ID mapped to venue name
                # Conference series ID mapped to venue name
                # Paper rank
                try :
                    row = line.split('\t')
                    if len(row) != nfields:
                        print "Drop an invalid record."
                        continue

                    papers.append(tuple(row))
                    npapers += 1

                    # Buffer to avoid DB accesses
                    if len(papers) == 10000:
                        db.insert(into=table_name, fields=fields, values=papers, ignore=True)
                        papers[:] = []   # Empty the list
                        print "%d processed." % npapers


                # If an exception occurs, there was no data to be
                # processed, so just skip it
                except:
                    pass

            db.insert(into=table_name, fields=fields, values=papers, ignore=True)
            print "totally %d processed." % npapers

    except Exception, e:
        print e
        sys.exit()

    f.close()


def import_paper_author_affils(file_path, table_name='paper_author_affils'):
    table_description = ['id INT NOT NULL AUTO_INCREMENT',
                        'paper_id VARCHAR(30) NOT NULL',
                        'author_id VARCHAR(30)',
                        'affil_id VARCHAR(30)',
                        # 'affil_name VARCHAR(1200)', # too long but useless
                        'normal_affil_name VARCHAR(300)',
                        'author_seq_num SMALLINT UNSIGNED',
                        'PRIMARY KEY (id)',
                        'KEY (paper_id)',
                        'KEY (author_id)',
                        'KEY (affil_id)']
    db.create_table(table_name, table_description)

    fields = ['paper_id', 'author_id', 'affil_id', 'normal_affil_name', 'author_seq_num']

    npaper_author_affils = 0
    paper_author_affils = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip('\r\n').strip(' ')
                if line == '':
                    continue

                # Paper ID
                # Author ID
                # Affiliation ID
                # Original affiliation name
                # Normalized affiliation name
                # Author sequence number
                try :
                    row = line.split('\t')
                    if row[-1] != '': row[-1] = int(row[-1])
                    del row[3] # delete Original affiliation name

                    paper_author_affils.append(tuple(row))
                    npaper_author_affils += 1

                    if len(paper_author_affils) == 500000:
                        db.insert(into=table_name, fields=fields, values=paper_author_affils, ignore=True)
                        paper_author_affils[:] = []   # Empty the list
                        print "%d processed." % npaper_author_affils


                # If an exception occurs, there was no data to be
                # processed, so just skip it
                except:
                    pass

            db.insert(into=table_name, fields=fields, values=paper_author_affils, ignore=True)
            print "totally %d processed." % npaper_author_affils

    except Exception, e:
        print e
        sys.exit()

    f.close()


def import_paper_urls(file_path, table_name='paper_urls'):
    table_description = ['paper_id VARCHAR(30) NOT NULL',
                        'url VARCHAR(200)',
                        'PRIMARY KEY (paper_id)']
    db.create_table(table_name, table_description)

    fields = ["paper_id", "url"]
    npaper_urls = 0
    paper_urls = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip('\r\n').strip(' ')
                if line == '':
                    continue

                # Paper ID
                # URL
                try :
                    paper_id, url = line.split('\t')
                    paper_urls.append((paper_id, url))
                    npaper_urls += 1

                    if len(paper_urls) == 100000:
                        db.insert(into=table_name, fields=fields, values=paper_urls, ignore=True)
                        paper_urls[:] = []   # Empty the list
                        print "%d processed." % npaper_urls


                # If an exception occurs, there was no data to be
                # processed, so just skip it
                except:
                    pass

            db.insert(into=table_name, fields=fields, values=paper_urls, ignore=True)
            print "totally %d processed." % npaper_urls

    except Exception, e:
        print e
        sys.exit()

    f.close()


def import_field_of_study(file_path, table_name='field_of_study'):
    table_description = ['id VARCHAR(30) NOT NULL',
                        'name VARCHAR(200) NOT NULL',
                        'PRIMARY KEY (id)']
    db.create_table(table_name, table_description)

    nfields_of_study = 0
    fields_of_study = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip('\r\n').strip(' ')
                if line == '':
                    continue

                # Field of study ID
                # Field of study name
                try :
                    id, name = line.split('\t')
                    fields_of_study.append((id, name))
                    nfields_of_study += 1

                    if len(fields_of_study) == 100000:
                        db.insert(into=table_name, fields=["id", "name"], values=fields_of_study, ignore=True)
                        fields_of_study[:] = []   # Empty the list
                        print "%d processed." % nfields_of_study


                # If an exception occurs, there was no data to be
                # processed, so just skip it
                except:
                    pass

            db.insert(into=table_name, fields=["id", "name"], values=fields_of_study, ignore=True)
            print "totally %d processed." % nfields_of_study

    except Exception, e:
        print e
        sys.exit()

    f.close()


def import_fields_of_study_hierarchy(file_path, table_name='fields_of_study_hierarchy'):
    table_description = ['id INT NOT NULL AUTO_INCREMENT',
                        'child_id VARCHAR(30) NOT NULL',
                        'child_level VARCHAR(2) NOT NULL',
                        'parent_id VARCHAR(30) NOT NULL',
                        'parent_level VARCHAR(2) NOT NULL',
                        'confidence FLOAT(18,17) NOT NULL',
                        'PRIMARY KEY (id)',
                        'KEY (child_id)',
                        'KEY (parent_id)']
    db.create_table(table_name, table_description)

    fields = ["child_id", "child_level", "parent_id", "parent_level", "confidence"]
    nfields_of_study_hier = 0
    fields_of_study_hier = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip('\r\n').strip(' ')
                if line == '':
                    continue

                # Child field of study ID
                # Child field of study level
                # Parent field of study ID
                # Parent field of study level
                # Confidence
                try :
                    row = line.split('\t')
                    fields_of_study_hier.append(tuple(row))
                    nfields_of_study_hier += 1

                    if len(fields_of_study_hier) == 100000:
                        db.insert(into=table_name, fields=fields, values=fields_of_study_hier, ignore=True)
                        fields_of_study_hier[:] = []   # Empty the list
                        print "%d processed." % nfields_of_study_hier


                # If an exception occurs, there was no data to be
                # processed, so just skip it
                except:
                    pass

            db.insert(into=table_name, fields=fields, values=fields_of_study_hier, ignore=True)
            print "totally %d processed." % nfields_of_study_hier

    except Exception, e:
        print e
        sys.exit()

    f.close()


def get_conf_docs(conf_id=None, year=None):
    """
    Get pub records from selected conferences in selected years in papers dataset.
    """

    # Check parameter types
    if isinstance(conf_id, basestring):
        conf_id_str = "('%s')"%str(conf_id)

    elif hasattr(conf_id, '__iter__'):
        if len(conf_id) == 0:
            return []
        elif len(conf_id) == 1:
            conf_id_str = "(%s)" % conf_id[0]
        else:
            conf_id_str = str(tuple(conf_id))

    else:
        raise TypeError("Parameter 'conf_id' is of unsupported type. String or iterable needed.")

    if isinstance(year, basestring):
        year_str = "(%s)"%str(year)

    elif hasattr(year, '__iter__'):
        if len(year) == 0:
            return []
        elif len(year) == 1:
            year_str = "(%s)" % year[0]
        else:
            year_str = str(tuple(year))

    else:
        raise TypeError("Parameter 'year' is of unsupported type. String or iterable needed.")


    year_cond = "year IN %s"%year_str if year else ""
    conf_id_cond = "conf_id IN %s"%conf_id_str if conf_id else ""

    if year_cond != '' and conf_id_cond != '':
        where_cond = '%s AND %s'%(year_cond, conf_id_cond)
    elif year_cond == '' and conf_id_cond != '':
        where_cond = conf_id_cond
    elif year_cond != '' and conf_id_cond == '':
        where_cond = year_cond
    else:
        where_cond = None

    rst = db.select('id', 'papers', where=where_cond)

    return rst



def get_selected_docs(conf_name=None, year=None):
    """
    Get pub records from selected conferences in selected years.
    """

    # Check parameter types
    if isinstance(conf_name, basestring):
        conf_name_str = "('%s')"%str(conf_name)

    elif hasattr(conf_name, '__iter__'):
        if len(conf_name) == 0:
            return []
        elif len(conf_name) == 1:
            conf_name_str = "(%s)" % conf_name[0]
        else:
            conf_name_str = str(tuple(conf_name))

    else:
        raise TypeError("Parameter 'conf_name' is of unsupported type. String or iterable needed.")

    if isinstance(year, basestring):
        year_str = "(%s)"%str(year)

    elif hasattr(year, '__iter__'):
        if len(year) == 0:
            return []
        elif len(year) == 1:
            year_str = "(%s)" % year[0]
        else:
            year_str = str(tuple(year))

    else:
        raise TypeError("Parameter 'year' is of unsupported type. String or iterable needed.")


    year_cond = "year IN %s"%year_str if year else ""
    conf_name_cond = "venue_abbr_name IN %s"%conf_name_str if conf_name else ""

    if year_cond != '' and conf_name_cond != '':
        where_cond = '%s AND %s'%(year_cond, conf_name_cond)
    elif year_cond == '' and conf_name_cond != '':
        where_cond = conf_name_cond
    elif year_cond != '' and conf_name_cond == '':
        where_cond = year_cond
    else:
        where_cond = None

    rst = db.select(['id', 'year'], 'selected_papers', where=where_cond)

    return rst


def get_conf_pubs(conf_id=None, year=None):
    """
    Get pub records from selected conferences in selected years in papers dataset.
    """

    # Check parameter types
    if isinstance(conf_id, basestring):
        conf_id_str = "('%s')"%str(conf_id)

    elif hasattr(conf_id, '__iter__'):
        if len(conf_id) == 0:
            return []
        elif len(conf_id) == 1:
            conf_id_str = "(%s)" % conf_id[0]
        else:
            conf_id_str = str(tuple(conf_id))

    else:
        raise TypeError("Parameter 'conf_id' is of unsupported type. String or iterable needed.")

    if isinstance(year, basestring):
        year_str = "(%s)"%str(year)

    elif hasattr(year, '__iter__'):
        if len(year) == 0:
            return []
        elif len(year) == 1:
            year_str = "(%s)" % year[0]
        else:
            year_str = str(tuple(year))

    else:
        raise TypeError("Parameter 'year' is of unsupported type. String or iterable needed.")


    year_cond = "papers.year IN %s"%year_str if year else ""
    conf_id_cond = "papers.conf_id IN %s"%conf_id_str if conf_id else ""

    if year_cond != '' and conf_id_cond != '':
        where_cond = '%s AND %s'%(year_cond, conf_id_cond)
    elif year_cond == '' and conf_id_cond != '':
        where_cond = conf_id_cond
    elif year_cond != '' and conf_id_cond == '':
        where_cond = year_cond
    else:
        where_cond = None

    rst = db.select(['papers.id', 'paper_author_affils.author_id', 'paper_author_affils.affil_id', 'papers.year'], \
            ['papers', 'paper_author_affils'], join_on=['id', 'paper_id'], \
            where=where_cond)

    import pdb;pdb.set_trace()
    # re-pack data to this format: {paper_id: {author_id:[affil_id,],},}
    pub_records = defaultdict()
    for paper_id, author_id, affil_id, year in rst:
        if pub_records.has_key(paper_id):
            if pub_records[paper_id]['author'].has_key(author_id):
                pub_records[paper_id]['author'][author_id].append(affil_id)
            else:
                pub_records[paper_id]['author'][author_id] = [affil_id]
        else:
            pub_records[paper_id] = {'author': {author_id: [affil_id]}, 'year':int(year)}

    return pub_records



def retrieve_affils_by_author_papers(author_id, paper_id, table_name='dblp'):
    if table_name == 'csx':
        try:
            author_name = db.select("name", "authors", where="id='%s'"%author_id, limit=1)[0].strip('\r\n ')

            paper_table = "selected_papers" # temporary setting, to be improved
            paper_title = db.select("title", paper_table, where="id='%s'"%paper_id, limit=1)[0].strip('\r\n ')

            if not author_name or not paper_title:
                return []

            # transfer to the external paper id
            transfered_paper_id = db.select("id", "%s_papers"%table_name, \
                    where="title REGEXP '[[:<:]]%s[[:>:]]'"%paper_title, limit=1)

            if not transfered_paper_id:
                return []
            print "transfer paper_id: %s"%transfered_paper_id[0]

            affil_names = db.select("affil", "%s_paper_author_affils"%table_name, \
                    where="paperid='%s' AND name='%s'"%(transfered_paper_id[0], author_name))

            if not affil_names:
                return []

            affil_ids = set()
            for each_affil_name in affil_names:
                # import pdb;pdb.set_trace()
                name_of_affil = reg_parse(each_affil_name)
                if name_of_affil:
                    affil = db.select("id", "affils", where="name REGEXP '[[:<:]]%s[[:>:]]'"%name_of_affil, limit=1)
                    if affil:
                        affil_ids.add(affil[0])
            return list(affil_ids)

        except Exception, e:
            print e
            # import pdb;pdb.set_trace()
            return []


    elif table_name == 'dblp':
        try:
            author_name = db.select("name", "authors", where="id='%s'"%author_id, limit=1)
            if not author_name:
                return []
            else:
                author_name = author_name[0].strip('\r\n ')
            paper_title = db.select("title", "selected_papers", where="id='%s'"%paper_id, limit=1)

            if not paper_title:
                paper_title = db.select("title", "expanded_conf_papers2", where="id='%s'"%paper_id, limit=1)

            if not paper_title:
                return []
            else:
                paper_title = paper_title[0].strip('\r\n. ')

            affil_names = db.select("dblp_auth_affil.affil_name", ["dblp_auth_affil", "dblp_auth_pub"], join_on=["dblp_key", "dblp_key"],
                    where="(dblp_auth_affil.name='%s' OR dblp_auth_affil.other_names REGEXP '[[:<:]]%s[[:>:]]') AND dblp_auth_pub.pub_title REGEXP '[[:<:]]%s[[:>:]]'"\
                            %(author_name, author_name, paper_title))

            if not affil_names:
                return []

            affil_ids = set()
            for each_affil_name in set(affil_names):
                # import pdb;pdb.set_trace()
                name_of_affil = reg_parse(each_affil_name)
                print "retrieved (by paper-author) affil name: %s"%name_of_affil
                if name_of_affil:
                    affil = db.select("id", "affils", where="name REGEXP '[[:<:]]%s[[:>:]]'"%name_of_affil, limit=1)
                    if affil:
                        affil_ids.add(affil[0])
                        print "retrieved (by paper-author) affil id: %s"%affil[0]
            return list(affil_ids)

        except Exception, e:
            print e
            # import pdb;pdb.set_trace()
            return []

    else:
        raise ValueError("Unknown value of parameter table_name. Parameter table_name should either be 'dblp' or 'csx'.")

# retrieve affils based on author name
# if paper_id is set, we use paper_id to confirm the exact author
# when there are multiple author-affil pairs
# which may be caused by shared author names
def retrieve_affils_by_authors(author_id, table_name='csx', paper_id=None):
    """
    we check csx_paper_author_affils table and do string matching which is knotty.
    """

    n_author_recall = 0
    author_name = db.select("name", "authors", where="id='%s'"%author_id, limit=1)[0].strip('\r\n ')

    # print author_name
    table_dblp_auth_affil = 'dblp_auth_affil'
    table_dblp_auth_pub = 'dblp_auth_pub'

    if table_name == 'all':
        affil_names1 = db.select("affil_name", table_dblp_auth_affil, where="name='%s' OR other_names REGEXP '[[:<:]]%s[[:>:]]'"%(author_name, author_name))
        affil_names2 = db.select("affil", "csx_paper_author_affils", where="name='%s'"%author_name)
        affil_names1.extend(affil_names2)
        affil_names = affil_names1
    elif table_name == 'dblp':
        affil_names = db.select("affil_name", table_dblp_auth_affil, where="name='%s' OR other_names REGEXP '[[:<:]]%s[[:>:]]'"%(author_name, author_name))
        if affil_names:
            n_author_recall += 1
            # import pdb;pdb.set_trace()
        if len(affil_names) > 1 and paper_id:
            import pdb;pdb.set_trace()
            affil_names2 = retrieve_affils_by_author_papers(author_id, paper_id, table_name='dblp')
            if affil_names2:
                affil_names = affil_names2
                print 'succeeded to retrieve affils by paper-author.'
            else:
                affil_names = affil_names[:1]
                print 'failed to retrieve affils by paper-author.'


    elif table_name == 'csx':
        affil_names = db.select("affil", "csx_paper_author_affils", where="name='%s'"%author_name)
    else:
        raise ValueError("Unknown table_name. Parameter table_name must be either 'csx' or 'dblp'.")
    match_affil_ids = set()


    for each_affil_name in affil_names:
        if not each_affil_name:
            continue

        affil_tokens = each_affil_name.lower().replace(",", " ").replace(";", " ").replace(".", " ").replace("-", " ").split()

        match_flag = False

        # import pdb;pdb.set_trace()
        # check if it's a company
        for com in affil_companies:
            if isinstance(com, str):
                if com in affil_tokens:
                    name_of_affil = com
                    match_flag = True
                    break
            else:
                flag = True
                for each in com:
                    if not each in affil_tokens:
                        flag = False
                        break
                if flag:
                    name_of_affil = ' '.join(com)
                    match_flag = True
                    break

        if match_flag: # match a company
            # print 'company'
            affil_ids = db.select("id", "affils", where="name REGEXP '[[:<:]]%s[[:>:]]'"%name_of_affil)

            match_affil_ids.update(set(affil_ids))
        else:
            # check if it's an academic institute

            # 1) first, check abbr.
            for abbr, univ in affil_univs.iteritems():
                abbr_l = abbr.split()
                flag = True
                for each in abbr_l:
                    if not each in affil_tokens:
                        flag = False
                        break
                if flag:
                    name_of_affil = univ
                    match_flag = True
                    break

            if not match_flag:
                # 2) then, check full name
                name_of_affil = reg_parse(each_affil_name) # parse affil names
                if not name_of_affil:
                    # if n_author_recall > 0:
                        # print name_of_affil
                        # import pdb;pdb.set_trace()
                    continue

                match_flag = True

            if match_flag:
                # print 'univ.'
                # special transform
                if special_transform.has_key(name_of_affil):
                    name_of_affil = special_transform[name_of_affil]

                try:
                    affil_ids = db.select("id", "affils", where="name REGEXP '[[:<:]]%s[[:>:]]'"%name_of_affil)
                except Exception,e:
                    print e
                if affil_ids:
                    match_affil_ids.update(set(affil_ids))
            else:
                # print 'yes name, no id'
                # import pdb;pdb.set_trace()
                pass
    # print "n_author_recall: %s"%n_author_recall
    return match_affil_ids, n_author_recall

# helper methods

def import_more_conf_pubs(conf_name):
    table_description = ['paper_id VARCHAR(30) NOT NULL',
                        'title VARCHAR(300) NOT NULL',
                        'year SMALLINT',
                        'conf_id VARCHAR(30) NOT NULL',
                        'PRIMARY KEY (paper_id)',
                        'KEY (conf_id)'
                        ]
    table_name = 'expanded_conf_papers2'
    db.create_table(table_name, table_description)
    fields = ['paper_id', 'title', 'year', 'conf_id']

    if conf_name == 'SIGIR':
        from datasets.expanded_SIGIR import py_sigir
        db.insert(into=table_name, fields=fields, values=py_sigir, ignore=True)
    elif conf_name == 'SIGMOD':
        from datasets.expanded_SIGMOD import py_sigmod
        db.insert(into=table_name, fields=fields, values=py_sigmod, ignore=True)
    elif conf_name == 'SIGCOMM':
        from datasets.expanded_SIGCOMM import py_sigcomm
        db.insert(into=table_name, fields=fields, values=py_sigcomm, ignore=True)

# for SimpleSearcher
def get_selected_expand_pubs(conf, year, _type="selected"):
    """
    Get expanded pub records from selected conferences in selected years.
    """

    # Check parameter types
    if isinstance(conf, basestring):
        conf_str = "('%s')"%str(conf)

    elif hasattr(conf, '__iter__'): # length of tuple should be larger than 1, otherwise use string
        conf_str = str(tuple(conf))

    else:
        raise TypeError("Parameter 'conf_id' is of unsupported type. String or iterable needed.")

    if isinstance(year, basestring):
        year_str = "(%s)"%str(year)

    elif hasattr(year, '__iter__'): # length of tuple should be larger than 1, otherwise use string
        year_str = str(tuple(year))

    else:
        raise TypeError("Parameter 'year' is of unsupported type. String or iterable needed.")

    if _type == 'selected':
        table_name = "%s_papers" % _type
        col_id_name = "id"
        conf_cond = "%s.venue_abbr_name IN %s"%(table_name, conf_str) if conf_str else ""

    elif _type == 'expanded':
        table_name = "%s_conf_papers" % _type
        col_id_name = "paper_id"
        conf_cond = "%s.conf_id IN %s"%(table_name, conf_str) if conf_str else ""

    else:
        raise ValueError("Unknown parameter type. parameter type should either be 'selected' or 'expanded'")

    year_cond = "%s.year IN %s"%(table_name, year_str) if year else ""

    if year_cond != '' and conf_cond != '':
        where_cond = '%s AND %s'%(year_cond, conf_cond)
    elif year_cond == '' and conf_cond != '':
        where_cond = conf_cond
    elif year_cond != '' and conf_cond == '':
        where_cond = year_cond
    else:
        where_cond = None

    rst = db.select(['%s.%s'%(table_name, col_id_name), 'paper_author_affils.author_id', 'paper_author_affils.affil_id', '%s.year'%table_name], \
            [table_name, 'paper_author_affils'], join_on=[col_id_name, 'paper_id'], \
            where=where_cond)


    # re-pack data to this format: {paper_id: {author_id:[affil_id,],},}
    count = 0
    get_affil_count = 0
    author = set()
    retrieved_paper_authors = set()
    pub_records = defaultdict()
    for paper_id, author_id, affil_id, year in rst:
        if affil_id == '' and not author_id in author:
            count += 1
            author.add(author_id)
        if not affil_id:
            if not (paper_id, author_id) in retrieved_paper_authors:
                # print "author id: %s"%author_id
                # print "paper id: %s"%paper_id
                # import pdb;pdb.set_trace()
                # retrieved_affil_ids = None # turn off
                retrieved_affil_ids, flag = retrieve_affils_by_authors(author_id, table_name='dblp', paper_id=paper_id)
                if flag == 1:
                    get_affil_count += 1
                # retrieved_affil_ids = retrieve_affils_by_author_papers(author_id, paper_id, table_name='csx')
                if retrieved_affil_ids:
                    retrieved_paper_authors.add((paper_id, author_id))
                    # print "retrieved", (paper_id, author_id, retrieved_affil_ids)
                    #############
                    # update db
                    #############
                    # to do

                    affil_id = set(retrieved_affil_ids)
                else:
                    # skip this record
                    continue
            else:
                continue

            # continue

        else:
            affil_id = set([affil_id])

        author.add(author_id)
        if pub_records.has_key(paper_id):
            if pub_records[paper_id]['author'].has_key(author_id):
                pub_records[paper_id]['author'][author_id].update(affil_id)
            else:
                pub_records[paper_id]['author'][author_id] = affil_id
        else:
            pub_records[paper_id] = {'author': {author_id: affil_id}, 'year':int(year)}
    print "missing author: %s/%s"%(count, len(author))
    print "get_affil_count: %s"%get_affil_count
    return pub_records


def reg_parse(affil_name):
    normal_affil_name = affil_name.title().replace('Univ.', 'University')\
                .replace('Umversity', 'University').replace('Universit', 'University')\
                .replace('Universityy', 'University') # low-prob case

    # try matching university and academy
    rst = univ_academy_prog.search(normal_affil_name)
    if not rst:
        rst = institute_prog.search(normal_affil_name)
        if not rst:
            rst = college_prog.search(normal_affil_name)
            if not rst:
                rst = univ_academy_prog2.search(normal_affil_name)
                if not rst:
                    rst = institute_prog2.search(normal_affil_name)
                    if not rst:
                        rst = college_prog2.search(normal_affil_name)
                        if not rst:
                            # print each_affil_name
                            # import pdb;pdb.set_trace()
                            return

    name_of_affil = rst.group(0).replace('-', '').replace('The', '')\
                            .replace('(', '').replace(')', '').strip()
    return name_of_affil



if __name__ == '__main__':
    # import_papers(config.DATA + 'Papers/Papers.txt')
    # import_authors('/Volumes/Mixed-Data/data/MAG/Authors/Authors.txt')
    import_more_conf_pubs('SIGIR')
    import_more_conf_pubs('SIGMOD')
    import_more_conf_pubs('SIGCOMM')
    # get_conf_pubs(conf_id='460A7036', year=range(2000,2011))
    # pass
