'''
Created on Mar. 10, 2016

@author: hugo
'''
from collections import defaultdict
from mymysql.mymysql import MyMySQL
# from utils import progress, plot
import sys
import config
import chardet
# from pylucene import Index, DocField


db = MyMySQL(config.DB_NAME, user=config.DB_USER, passwd=config.DB_PASSWD)

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
                line = line.strip(' ')
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
                line = line.strip(' ')
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
                line = line.strip(' ')
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
                line = line.strip(' ')
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
                line = line.strip(' ') # there are missing values in this dataset, keep \t sign for locating
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
                line = line.strip(' ')
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
                line = line.strip(' ')
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
                line = line.strip(' ')
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
                line = line.strip(' ')
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

    npapers = 0
    papers = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip(' ')
                if line == '':
                    continue

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
                line = line.strip(' ')
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

                    if len(paper_author_affils) == 100000:
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
                line = line.strip(' ')
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
                line = line.strip(' ')
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
                line = line.strip(' ')
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


if __name__ == '__main__':
    # import_papers(config.DATA + 'Papers/Papers.txt')
    import_paper_author_affils('/Volumes/Mixed-Data/data/MAG/PaperAuthorAffiliations/PaperAuthorAffiliations.txt')
    # pass
