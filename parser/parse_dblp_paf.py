'''
Created on Mar 28, 2016

@author: hugo
An event-driven parser which parses paper-author-affil relationship
'''

import requests
import xml.sax
import xml.dom.minidom
import lxml.html
import lxml.etree
from mymysql.mymysql import MyMySQL
from datasets.affil_names import url_keywords
import config
import subprocess
import sys
import re, time
import chardet


from requests.adapters import HTTPAdapter

s = requests.Session()
s.mount('https://', HTTPAdapter(max_retries=5))
s.mount('http://', HTTPAdapter(max_retries=5))


# For test
# Begin
start_author = 0
start_dblp_key = "homepages/90/5406"
# start_flag = False
# End


total_lineno = None
BASE_URL = 'http://dblp.uni-trier.de/'

db = MyMySQL(config.DB_NAME, user=config.DB_USER, passwd=config.DB_PASSWD)
auth_affil_bulk = set()
auth_pub_bulk = set()

url_pattern = '^(http[s]*://)([.a-z^/][^/]+[.a-z^/])?'
url_prog = re.compile(url_pattern)

homepage_pattern1 = 'http://dblp.uni-trier.de/pers/hd/(.)*"\s+itemprop="url"'
homepage_pattern2 = 'hd/(\S)+"'
homepage_prog1 = re.compile(homepage_pattern1)
homepage_prog2 = re.compile(homepage_pattern2)
# alltags = set()




# regexp patterns
univ_academy_pattern = '([A-Z][^\\s,.:;>]+[.]?\\s[(]?)*(University|Universidade|Academy|Council)[^,<;:.\\d]*(?=,|\\d|;|<|:|-|\\.)'
institute_pattern = '([A-Z][^\\s,.:;>]+[.]?\\s[(]?)*(Institute)[^,<;:.\\d]*(?=,|\\d|;|<|:|-|\\.)'
college_pattern = '([A-Z][^\\s,.:;>]+[.]?\\s[(]?)*(College|Centre|Center)[^,<;:.\\d]*(?=,|\\d|;|<|:|-|\\.)'

univ_academy_pattern2 = '([A-Z][^\\s,.:;>]+[.]?\\s[(]?)*(University|Universidade|Academy|Council)[^,<;:.\\d]*$'
institute_pattern2 = '([A-Z][^\\s,.:;>]+[.]?\\s[(]?)*(Institute)[^,<;:.\\d]*$'
college_pattern2 = '([A-Z][^\\s,.:;>]+[.]?\\s[(]?)*(College|Centre|Center)[^,<;:.\\d]*$'


univ_academy_prog = re.compile(univ_academy_pattern)
institute_prog = re.compile(institute_pattern)
college_prog = re.compile(college_pattern)

univ_academy_prog2 = re.compile(univ_academy_pattern2)
institute_prog2 = re.compile(institute_pattern2)
college_prog2 = re.compile(college_pattern2)



# All tags we have in dblp.xml
# [u'www', u'isbn', u'ee', u'series', u'number', u'month', u'mastersthesis', u'year', u'sub', u'title', u'incollection', u'booktitle', u'note', u'book', u'editor', u'sup', u'cite', u'journal', u'volume', u'address', u'cdrom', u'article', u'pages', u'crossref', u'chapter', u'publisher', u'school', u'phdthesis', u'dblp', u'inproceedings', u'i', u'author', u'url', u'proceedings', u'tt']
class DBLPHandler(xml.sax.ContentHandler):
    def __init__(self, locator, table_auth_affil, fields_auth_affil, table_auth_pub, fields_auth_pub):
        self.loc = locator

        self.table_auth_affil = table_auth_affil
        self.fields_auth_affil = fields_auth_affil

        self.table_auth_pub = table_auth_pub
        self.fields_auth_pub = fields_auth_pub

        self.setDocumentLocator(self.loc)
        self.CurrentData = ""
        self.valid = False # check if www tag
        self.is_affil = False # check if valid (having affiliation attr) note tag
        self.url = "" # we use urls to retrieve affiliations
        self.dblp_key = "" # dblp key for each author
        self.authors = []
        self.affils = set()
        # self.auth_affil_bulk = set()
        self.valid_count = 0 # num of homepage records as well as total number of authors including those we failed to find affils
        self.author_count = 0 # num of valid (i.e., having affils) authors
        self.auth_pub_count = 0
        self.count = 0 # num of tags
        self.nrows_auth_affil = 0
        self.start_flag = False
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
            self.dblp_key = attr["key"]

        elif self.valid and tag == "note" and attr.has_key("type") \
                and attr["type"] == "affiliation":
            # affiliation
            self.is_affil = True

        # alltags.add(tag)



    # Call when an elements ends
    def endElement(self, tag):
        if self.valid and tag == "www":

            self.valid = False # reset flag

            if self.dblp_key == start_dblp_key:
                self.start_flag = True
                print "Start to get authors, affils, pubs..."
                if self.authors:
                    self.authors[:] = []
                if self.affils:
                    self.affils.clear()
                self.url = ""
                self.dblp_key = ""
                self.is_affil = False
                return

            # for test
            if not self.start_flag:
                if self.authors:
                    self.authors[:] = []
                if self.affils:
                    self.affils.clear()
                self.url = ""
                self.dblp_key = ""
                self.is_affil = False
                return




            # pack data
            if not self.affils and self.url: # retrieve affils based on urls
                # import pdb;pdb.set_trace()
                self.affils = retrieve_affils_by_urls(self.url)
                if not self.affils:
                    self.affils = retrieve_affils_by_urls2(self.url)
                if not self.affils:
                    print self.url#, self.affils


            if self.affils:
                affil_names = self.affils

                if self.authors:

                    author_name = self.authors[0]
                    pubs = set()
                    if self.nrows_auth_affil >= start_author:
                        pubs = get_pubs_by_authors(author_name, self.dblp_key) # passes author_name before cleanning it
                    if pubs:
                        self.auth_pub_count += 1

                    author_name = re.sub(" \d+", " ", author_name) # remove digits at the end of the string
                    other_names = self.authors[1:]
                    # write to db
                    # print "author name: %s" % author_name
                    # print "other names: %s" % other_names
                    # print "affil names: %s" % affil_name
                    # print

                    # 1) stores pubs
                    auth_pubs = set()
                    for each in pubs:
                        auth_pubs.add((self.dblp_key, each))

                    auth_pub_bulk.update(auth_pubs)


                    # 2) one author may have multiple affils, we store all of them
                    auth_affils = set()
                    for each in affil_names:
                        auth_affils.add((self.dblp_key, author_name, '/'.join([x for x in other_names]), each))

                    auth_affil_bulk.update(auth_affils)

                    # table_auth_affil, fields_auth_affil, table_auth_pub, fields_auth_pub
                    if len(auth_affil_bulk) % 500 == 0:
                        if self.nrows_auth_affil >= start_author:
                            try:
                                db.insert(into=self.table_auth_affil, fields=self.fields_auth_affil, values=list(auth_affil_bulk), ignore=True)
                            except Exception,e:
                                print e
                                import pdb;pdb.set_trace()
                            # pass
                        auth_affil_bulk.clear()
                        self.nrows_auth_affil += 500



                    if len(auth_affil_bulk) % 500 == 0:
                        if self.nrows_auth_affil >= start_author:
                            db.insert(into=self.table_auth_pub, fields=self.fields_auth_pub, values=list(auth_pub_bulk), ignore=True)
                        auth_pub_bulk.clear()


                    self.author_count += 1
                    if self.author_count % 1000 == 0:
                        print "%s valid (having affils) authors processed." % self.author_count


            if self.authors:
                self.authors[:] = []
            if self.affils:
                self.affils.clear()
            self.url = ""
            self.dblp_key = ""

            if self.valid_count % 1000 == 0:
                print "%s homepages processed."%self.valid_count


        elif self.is_affil and tag == "note":
            self.is_affil = False

        # elif self.CurrentTag == "author":
        #     print "Author:", self.author
        # elif self.CurrentTag == "note":
        #     print "Note:", self.note
        # elif self.CurrentTag == "title":
        #     print "Title:", self.title
        # elif self.CurrentTag == "url":
        #     print "URL:", self.url

        self.CurrentTag = ""



    # Call when a character is read
    def characters(self, content):
        if self.valid and self.CurrentTag == "author":
            self.authors.append(content.strip('\r\n').strip())

        elif self.is_affil and self.CurrentTag == "note":
            # some affil name in the tag is not quite clean (e.g., Yangzhou University, School of Mathematics,
            # but what we really want is Yangzhou University)
            # try cleaning it if that applies
            clean = reg_parse_affil_name(content.strip('\r\n '))
            affil_name = clean if clean else content.strip('\r\n ')
            self.affils.add(affil_name)

        elif self.valid and self.CurrentTag == "url":
            self.url = content.strip()
        # elif self.valid and self.CurrentTag == "title":
            # self.title = content

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


def retrieve_affils_by_urls2(url, search_engine='google'):
    """
    more sophisticated.
    """

    success_flag = False
    # 1) check domain page
    # rst = url_prog.search(url)
    rst = re.findall('^(http[s]*://)([.a-z^/][^/]+[.a-z^/])?', url)
    if not rst:
        return set()

    tokens = rst[0][1].split('.')
    if 'edu' in tokens:
        domain_url = rst[0][0] + 'www.' + '.'.join(tokens[(tokens.index('edu') - 1):])
    elif 'ac' in tokens:
        domain_url = rst[0][0] + 'www.' + '.'.join(tokens[(tokens.index('ac') - 1):])
    elif 're' in tokens:
        domain_url = rst[0][0] + 'www.' + '.'.join(tokens[(tokens.index('re') - 1):])
    else:
        domain_url = rst[0][0] + 'www.' + '.'.join(tokens[-2:])

    # resp = requests.get(rst.group(0))
    try:
        resp = requests.get(domain_url)
    except Exception, e:
        print e
        success_flag = False
        # import pdb;pdb.set_trace()
    else:

        if resp.status_code == 200:
            # import pdb;pdb.set_trace()
            title = re.findall('<title>(.+?)</title>', resp.content)
            if title:
                affil_names = reg_parse_affil_name(title[0])
                if affil_names:
                    # convert to utf-8 if necessary
                    affil_names = convert_to_unicode(affil_names)
                    return set([affil_names])
                else:
                    success_flag = False

            else:
                success_flag = False


        elif resp.status_code == 429:
            # Too Many Requests
            try:
                # print resp.headers
                twait = int(resp.headers["Retry-After"])
            except Exception, e:
                print e
                twait = 30 # default

            time.sleep(twait)
            print 'wait %s seconds. Retry...' % twait

            return retrieve_affils_by_urls2(url)

        else:
            # print "failed to retrieve the affil given the url: %s" % url

            # return set()
            success_flag = False

    if not success_flag:


        # 2) search the homepage
        # unfortunately, we can only do some rough matching currently
        # (i.e., we only match university, college or institute which might
        # overkill some info or obtain incorrect info.)

        # To avoid false positive, we only use this strategy on google scholar homepage
        # which often contains clean and simple data. We found that many of pages linked to
        # Wikipedia, but it's not easy to retrieve correct affilations from such pages for
        # some reason.
        if 'google' in tokens and 'scholar' in tokens:

            try:
                resp = requests.get(url)
            except Exception, e:
                print e
                success_flag = False
            else:
                if resp.status_code == 200:
                    affil_names = reg_parse_affil_name(resp.content)
                    if affil_names:
                        affil_names = convert_to_unicode(affil_names)
                        return set([affil_names])
                    else:
                        success_flag = False

                elif resp.status_code == 429:
                    # Too Many Requests
                    try:
                        # print resp.headers
                        twait = int(resp.headers["Retry-After"])
                    except Exception, e:
                        print e
                        twait = 30 # default

                    time.sleep(twait)
                    print 'wait %s seconds. Retry...' % twait

                    return retrieve_affils_by_urls2(url)

                else:
                    # print "failed to retrieve the affil given the url: %s" % url

                    # return set()
                    success_flag = False


        # 3) for some titles not written in English, we search the url with search engine
        if not success_flag:

            try:
                resp = requests.get('https://www.bing.com/search?q=%s'%domain_url)
            except Exception, e:
                print e
                return set()
            else:
                if resp.status_code == 200:
                    affil_names = reg_parse_affil_name(resp.content)
                    # findall()
                    return set([convert_to_unicode(affil_names)]) if affil_names else set()

                elif resp.status_code == 429:
                    # Too Many Requests
                    try:
                        # print resp.headers
                        twait = int(resp.headers["Retry-After"])
                    except Exception, e:
                        print e
                        twait = 30 # default

                    time.sleep(twait)
                    print 'wait %s seconds. Retry...' % twait

                    return retrieve_affils_by_urls2(url)

                else:
                    print "failed to retrieve the affil given the url: %s" % url

                    return set()



def retrieve_affils_by_urls(url):
    """
    using table matching.
    """

    rst = url_prog.search(url)
    if rst:
        url_tokens = rst.group(0).replace('/', '.').split('.')
        for k, v in url_keywords.iteritems():
            flag = True
            for each in k.split():
                if not each in url_tokens:
                    flag = False
                    break
            if flag:
                return set([v])

    return set()



# We cannot parse author_names with special characters currently
def get_pubs_by_authors(author_name, dblp_key):
    if not author_name:
        return set()

    # generate the url linked to the author's homepage
    rst = re.search(" \d+", author_name.strip()) # e.g., "Chen Li 0001"
    suffix = rst.group(0) if rst else ''

    name_token = author_name.replace(suffix,'').split(' ')

    last_name = ("%s%s"%(name_token[-1], suffix)).replace(' ', '_')
    # print last_name
    # e.g. "Chen Li 0001" -> "Li_0001:Chen", "Li-Chieh Chen" -> "Chen:Chieh=Li"
    # "Mohammed J. Zaki" -> "Zaki:Mohammed_J=", "Yunhua Li" -> "Li:Yunhua"
    # replace ' ' with '_', ('.', '-', ''') with '='
    url_key = (last_name + ':' + ('_'.join(name_token[:-1]))).replace('.', '=').replace('-', '=').replace("'", '=')
    # print url_key
    url = "%spers/xx/%s/%s.xml"%(BASE_URL, last_name[0].lower(), url_key)
    # print url

    # fetchs and parses the xml file
    resp = requests.get(url)

    if resp.status_code == 200:
        # import pdb;pdb.set_trace()
        pubs = parse_xml(resp.content, dblp_key)

        return pubs

    elif resp.status_code == 404:
        # The page is not found
        # Searches homepage directly by author name
        content = search_homepages(author_name)
        if not content:
            print "failed to find author: %s, dblp_key: %s (cannot find the homepage.)" % (author_name, dblp_key)
            return set()

        pubs = parse_xml(content, dblp_key)

        return pubs

    elif resp.status_code == 429:
        # Too Many Requests
        try:
            # print resp.headers
            twait = int(resp.headers["Retry-After"])
        except Exception, e:
            print e
            twait = 30 # default

        time.sleep(twait)
        print 'wait %s seconds. Retry...' % twait

        return get_pubs_by_authors(author_name, dblp_key)

    else:
        print "failed to find author: %s, dblp_key: %s (unknown reasons.)" % (author_name, dblp_key)

        return set()

def search_homepages(author_name):
    """
    get the exact homepage given a distinct author name.
    Users should make sure the author name is unique as
    the method only returns the first matching.
    """
    search_url = "%ssearch?q=%s" % (BASE_URL, '+'.join(author_name.split(' ')))
    resp = requests.get(search_url)
    # import pdb;pdb.set_trace()
    if resp.status_code == 200:
        # get the url of homepage
        # import pdb;pdb.set_trace()
        rst = homepage_prog1.search(resp.content)
        if rst:
            rst = homepage_prog2.search(rst.group(0))
            if rst:
                partital_url = rst.group(0).replace('hd/','').replace('"','')
                url = "%spers/xx/%s.xml" % (BASE_URL, partital_url)
                resp = requests.get(url)

                if resp.status_code == 200:
                    # import pdb;pdb.set_trace()
                    return resp.content

                elif resp.status_code == 404:
                    # The page is not found
                    # Searches homepage directly by author name
                    print "failed to find author: %s (cannot find the homepage.)" % (author_name)
                    return ''

                elif resp.status_code == 429:
                    # Too Many Requests
                    try:
                        # print resp.headers
                        twait = int(resp.headers["Retry-After"])
                    except Exception, e:
                        print e
                        twait = 30 # default

                    time.sleep(twait)
                    print 'wait %s seconds. Retry...' % twait

                    return search_homepages(author_name)

        print "failed to find author: %s (unknown reasons.)" % (author_name)

        return ''

    elif resp.status_code == 404:
        print "search API does not work currently."
        return ''

    elif resp.status_code == 429:
        # Too Many Requests
        try:
            # print resp.headers
            twait = int(resp.headers["Retry-After"])
        except Exception, e:
            print e
            twait = 30 # default

        time.sleep(twait)
        print 'wait %s seconds. Retry...' % twait

        return search_homepages(author_name)

    else:
        return ''



def parse_xml(content, dblp_key):
    """
    parses dblp homepages and returns the publications.
    dblp_key is used for checking.
    """

    try:
        DOMTree = xml.dom.minidom.parseString(content)
    except Exception, e:
        print e
        return set()

    dblp_person = DOMTree.documentElement
    # checks if this is exactlly the wanted homepage using dblp key
    person = dblp_person.getElementsByTagName("person")
    if person and person[0].hasAttribute("key"):
        if not dblp_key == person[0].getAttribute("key"): # not the right person
            # import pdb;pdb.set_trace()
            return set()

        else:
            # right person
            # fetchs pubs
            docs_set = set()
            # fetchs conference pubs
            pubs = dblp_person.getElementsByTagName("r")
            for each_pub in pubs:
                doc = each_pub.getElementsByTagName("inproceedings") # conference papers
                if doc:
                    title = doc[0].getElementsByTagName("title")
                    if title and title[0].firstChild.nodeValue:
                        try:
                            docs_set.add(title[0].firstChild.nodeValue.strip('. '))
                        except Exception, e:
                            print e
                            # import pdb;pdb.set_trace()
            return docs_set

    else:
        # import pdb;pdb.set_trace()
        return set()


def search_authors(author_name):
    """
    get all the homepages given an author name. author name is not necessarily unique.
    """
    # show all authors
    # import pdb;pdb.set_trace()
    search_url = "%ssearch?q=%s" % (BASE_URL, '+'.join(author_name.split(' ')))
    resp = requests.get(search_url)

    if resp.status_code == 200:
        # get the url of homepage
        # import pdb;pdb.set_trace()
        if re.search('<em>show all', resp.content):
            search_url = "%ssearch/author?q=%s" % (BASE_URL, '+'.join(author_name.split(' ')))
            root = lxml.html.parse(search_url)
        else:
            root = lxml.html.fromstring(resp.content)
        # check only exact matches
        if not root.xpath("//div[@id='completesearch-authors']/div/p[text()='Exact matches']"):
            if not root.xpath("//div[@id='completesearch-authors']/div/p[text()='Likely matches']"):
                return []

        authors = root.xpath("//div[@id='completesearch-authors']/div/ul[1]/li")

        homepages = []

        for each_author in authors:
            rst = homepage_prog1.search(lxml.etree.tostring(each_author))
            if rst:
                rst = homepage_prog2.search(rst.group(0))
                if rst:
                    partital_url = rst.group(0).replace('hd/','').replace('"','')
                    url = "%spers/xx/%s.xml" % (BASE_URL, partital_url)
                    resp = requests.get(url)

                    if resp.status_code == 200:
                        # import pdb;pdb.set_trace()
                        homepages.append(resp.content)

                    elif resp.status_code == 404:
                        # The page is not found
                        # Searches homepage directly by author name
                        # print "failed to find author: %s (cannot find the homepage.)" % (author_name)
                        pass

                    elif resp.status_code == 429:
                        # Too Many Requests
                        try:
                            # print resp.headers
                            twait = int(resp.headers["Retry-After"])
                        except Exception, e:
                            print e
                            twait = 30 # default

                        time.sleep(twait)
                        print 'wait %s seconds. Retry...' % twait

                        return search_authors(author_name)

            # print "failed to find author: %s (unknown reasons.)" % (author_name)
        return homepages

    elif resp.status_code == 404:
        print "search API does not work currently."
        return []

    elif resp.status_code == 429:
        # Too Many Requests
        try:
            # print resp.headers
            twait = int(resp.headers["Retry-After"])
        except Exception, e:
            print e
            twait = 30 # default

        time.sleep(twait)
        print 'wait %s seconds. Retry...' % twait

        return search_authors(author_name)

    else:
        return []

def search_affils_by_author_paper(author_name, paper_title):
    """
    get author's affils by author name and paper title
    """

    dblp_key = ''
    dblp_name = ''
    affils = set()
    # return all the authors having that name
    homepages = search_authors(author_name)

    for each in homepages:
        try:
            DOMTree = xml.dom.minidom.parseString(each)
        except Exception, e:
            print e
            return ""

        dblp_person = DOMTree.documentElement
        pub = [x for x in dblp_person.getElementsByTagName('title') if x.firstChild.nodeValue and x.firstChild.nodeValue.lower().strip('. ') == paper_title.lower().strip('. ')]
        if pub: # matched
            # get dblp_key
            person = dblp_person.getElementsByTagName("person")

            if person and person[0].hasAttribute("key"):
                dblp_key = person[0].getAttribute("key")
                dblp_name = dblp_person.getAttribute("name")

            ee = pub[0].parentNode.getElementsByTagName('ee')
            if ee:
                url = ee[0].firstChild.nodeValue
                print url
                if 'acm' in url.split('.'): # acm digital library
                    resp = requests.get(url)

                    if resp.status_code == 200:
                        try:
                            # import pdb;pdb.set_trace()
                            root = lxml.html.fromstring(resp.content)
                            href = root.xpath("//tr/td/a[text()='%s']"%author_name.title())

                            if not href: # not matched probably because the author name has special char (e.g., European char)
                                href = root.xpath("//tr/td/a[text()='%s']"%dblp_name.title())

                                # for each_part_of_name in author_name.title():
                                #     # href_set = set()
                                #     href = root.xpath("//tr/td/a[contains(text(), '%s')]"%each_part_of_name)
                                #     if len(href) == 1:
                                #         break
                                #     else:
                                #         href = []
                                #         continue

                            if not href:
                                break

                            tr = href[0].getparent().getparent()
                            # 1)
                            if tr.xpath("td/small"):
                                affil_name = tr.xpath("td/small")[0].text
                                affils.add(affil_name)
                            elif tr.xpath('td/a/small'):
                                affil_name = tr.xpath('td/a/small')[0].text
                                affils.add(affil_name)
                            # tmp = re.search('<a (\s*) href=(.*)%s</a>'%author_name.title(), resp.content).group()
                            # url = re.findall('href="(.*)"(\s*)title', tmp)[0][0]
                            resp = requests.get("http://dl.acm.org/%s" % href[0].attrib['href'])

                        except Exception, e:
                            print e
                            import pdb;pdb.set_trace()
                            break

                        else:
                            if resp.status_code == 200:
                                try:
                                    root = lxml.html.fromstring(resp.content)
                                    strong = root.xpath("//td/strong[text()='Affiliation history']")
                                    if strong:
                                        td = strong[0].getparent()

                                        for each in td.xpath("div/a"):
                                            affils.add(each.text)
                                        break

                                except Exception, e:
                                    print e
                                    # import pdb;pdb.set_trace()
                                    break

                            elif resp.status_code == 404:
                                print "search API does not work currently."
                                break

                            elif resp.status_code == 429:
                                # Too Many Requests
                                try:
                                    # print resp.headers
                                    twait = int(resp.headers["Retry-After"])
                                except Exception, e:
                                    print e
                                    twait = 30 # default

                                time.sleep(twait)
                                print 'wait %s seconds. Retry...' % twait

                                return search_affils_by_author_paper(author_name, paper_title)

                            else:
                                break

                    elif resp.status_code == 404:
                        print "search API does not work currently."
                        break

                    elif resp.status_code == 429:
                        # Too Many Requests
                        try:
                            # print resp.headers
                            twait = int(resp.headers["Retry-After"])
                        except Exception, e:
                            print e
                            twait = 30 # default

                        time.sleep(twait)
                        print 'wait %s seconds. Retry...' % twait

                        return search_affils_by_author_paper(author_name, paper_title)
                    else:
                        break


                elif 'doi' in url.split('.'):
                    resp = requests.get(url)

                    if resp.status_code == 200:
                        try:
                            # import pdb;pdb.set_trace()
                            root = lxml.html.fromstring(resp.content)
                            # it's stupid to split name and match it piece by piece, but it looks like we have no choice here
                            name = dblp_name.title().split()

                            candidates = set(root.xpath("//span[@class='AuthorName'][contains(text(),'%s')]"%name[0]))
                            for each in name[1:]:
                                candidates = candidates & set(root.xpath("//span[@class='AuthorName'][contains(text(),'%s')]"%each))

                            if not candidates:
                                break
                            try:
                                affil_name = list(candidates)[0].getparent().xpath("span")[1].xpath("span/span[@class='AuthorsName_affiliation']/span")[0].text
                                affils.add(affil_name)
                            except:
                                break

                        except Exception, e:
                            print e
                            import pdb;pdb.set_trace()
                            break

                    elif resp.status_code == 404:
                        print "search API does not work currently."
                        break

                    elif resp.status_code == 429:
                        # Too Many Requests
                        try:
                            # print resp.headers
                            twait = int(resp.headers["Retry-After"])
                        except Exception, e:
                            print e
                            twait = 30 # default

                        time.sleep(twait)
                        print 'wait %s seconds. Retry...' % twait

                        return search_affils_by_author_paper(author_name, paper_title)
                    else:
                        break

                else:
                    break
            else:
                break
        else:
            continue

    return dblp_key, affils



def get_dblp_key_by_author_paper(author_name, paper_title):
    """
    get author's dblp key by the name
    """

    # return all the authors having that name
    homepages = search_authors(author_name)
    for each in homepages:
        try:
            DOMTree = xml.dom.minidom.parseString(each)
        except Exception, e:
            print e
            return ''

        dblp_person = DOMTree.documentElement
        pub = [x for x in dblp_person.getElementsByTagName('title') if x.firstChild.nodeValue and x.firstChild.nodeValue.lower().strip('. ') == paper_title.lower().strip('. ')]
        if pub: # matched
            # get dblp_key
            person = dblp_person.getElementsByTagName("person")
            if person and person[0].hasAttribute("key"):
                dblp_key = person[0].getAttribute("key")
                if dblp_key:
                    return dblp_key
    return ''


def convert_to_unicode(_str):
    # import pdb;pdb.set_trace()
    try:
        coding = chardet.detect(_str)['encoding']
        _str = _str.decode(coding)
    except Exception, e:
        print e
    return _str


def reg_parse_affil_name(affil_name):
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
                            return ''

    name_of_affil = rst.group(0).replace('-', ' ').replace('The', '')\
                            .replace('(', '').replace(')', '').strip()
    name_of_affil = ' '.join(name_of_affil.split()) # a stupid way to merge multiple space chars to a single one
    return name_of_affil


if __name__ == "__main__":
    # try:
    #     in_file = sys.argv[1]
    # except Exception, e:
    #     print e
    #     sys.exit()

    # total_lineno = file_len(in_file)
    # print "total line num of the file: %s\n" % total_lineno

    # # db info
    # # table 1)
    # table_description_auth_affil = ['id INT NOT NULL AUTO_INCREMENT',
    #                     'dblp_key VARCHAR(200) NOT NULL',
    #                     'name VARCHAR(200) NOT NULL',
    #                     'other_names VARCHAR(1000)',
    #                     'affil_name VARCHAR(200)',
    #                     'PRIMARY KEY (id)',
    #                     'KEY (dblp_key)',
    #                     'KEY (name)']

    # table_auth_affil = "dblp_auth_affil_ext"
    # fields_auth_affil = ["dblp_key", "name", "other_names", "affil_name"]

    # # table 2)
    # table_description_auth_pub = ['id INT NOT NULL AUTO_INCREMENT',
    #                     'dblp_key VARCHAR(200) NOT NULL',
    #                     'pub_title VARCHAR(300) NOT NULL',
    #                     'PRIMARY KEY (id)',
    #                     'KEY (dblp_key)']

    # table_auth_pub = "dblp_auth_pub_ext"
    # fields_auth_pub = ["dblp_key", "pub_title"]

    # # create table
    # db.create_table(table_auth_affil, table_description_auth_affil, force=False)
    # db.create_table(table_auth_pub, table_description_auth_pub, force=False)


    # # create an XMLReader
    # parser = xml.sax.make_parser()
    # locator = xml.sax.expatreader.ExpatLocator(parser)
    # # turn off namepsaces
    # parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    # # override the default ContextHandler
    # Handler = DBLPHandler(locator, table_auth_affil, fields_auth_affil, table_auth_pub, fields_auth_pub)
    # parser.setContentHandler(Handler)


    # parser.parse(in_file)

    # # write remaining data into db
    # if auth_affil_bulk:
    #     db.insert(into=table_auth_affil, fields=fields_auth_affil, values=list(auth_affil_bulk), ignore=True)
    #     # pass

    # if auth_pub_bulk:
    #     db.insert(into=table_auth_pub, fields=fields_auth_pub, values=list(auth_pub_bulk), ignore=True)
    #     # pass

    # # print len(alltags)

    # # docs_set = get_pubs_by_authors(in_file, sys.argv[2])
    # # print docs_set
    # # print len(docs_set)

    # print "#################################"
    # print "############statistics###########"
    # print "#################################"
    # print "# of authors: %s" % Handler.valid_count
    # print "# of valid authors (having affils): %s" % Handler.author_count
    # print "# of valid authors we got pubs: %s" % Handler.auth_pub_count
    # print "It's done."
    # # dblp_keys = get_dblp_key_by_authnames("Chen Li")
    # # print dblp_keys
    # # print len(dblp_keys)


    # # affils = search_affils_by_author_paper("Jianshu Weng", 'Exploiting hybrid contexts for Tweet segmentation')
    # # print affils

    # dblp_key, affil_names = search_affils_by_author_paper('Francois Poulet', 'Integrating heterogeneous information within a social network for detecting communities.')
    dblp_key, affil_names = search_affils_by_author_paper('nguyenkhang pham', 'A comparison of different off-centered entropies to deal with class imbalance for decision trees')
    print dblp_key, affil_names
