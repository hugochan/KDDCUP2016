'''
Created on Jul 13, 2014

@author: luamct
'''

import zeno
from nltk.stem.porter import PorterStemmer
import os
import traceback
import sys
import re
from collections import Counter
# from relevance.topics import get_docs
from lxml import html
from subprocess import call, PIPE
from mymysql.mymysql import MyMySQL
import nltk
import config
import logging
import utils
from zeno.task_manager import NothingToProcessException


stemmer = PorterStemmer()
stopwords = set(nltk.corpus.stopwords.words('english'))

# This regex only matches 2 or more characters tokens
TOKEN_REGEX  = re.compile(r"(?u)\b\w\w+\b")
NUMBER_REGEX = re.compile("\d+$")
		

class MinimumTokensException(Exception) :
	pass


def fix_hyphens(text):
	return text.replace("-\n", "")


def match_any(text, names):
	for name in names: 
		if (text.find(name, 0, 20) >= 0) :
			return True

	return False


class Tokenizer() :

	def __init__(self) :

		# Zeno task manager
		self.tasks = zeno.TasksManager("tasks", 
																	 host=config.DB_HOST, 
																	 user=config.DB_USER,
																	 passwd=config.DB_PASSWD)

		# Database connection
		self.db = MyMySQL(db=config.DB_NAME,
											host=config.DB_HOST, 
											user=config.DB_USER,
											passwd=config.DB_PASSWD)

		# Logging configuration
		self.log = utils.config_logging('tokenizer', stream=sys.stdout, level=logging.DEBUG,
												format='%(asctime)s (%(name)s) [%(levelname)6s]: %(message)s',
												datefmt="%Y-%m-%d %H:%M:%S")

		self.MIN_TOKENS = 10

		# Create folders with non existing
		utils.ensure_folder(os.path.dirname(config.TOKENS_PATH))
		utils.ensure_folder(os.path.dirname(config.TOKENS_PATH_PARTS))


	def save_tokens(self, tokens, tok_file) :
		counter = Counter(tokens)
		with open(tok_file, 'w') as f :
# 			print >> f, (' '.join(tokens)).encode("utf-8")
			lines = ["%s %d" % (token, count) for (token, count) in counter.items()]
			print >> f, '\n'.join(lines).encode("UTF-8")


	def get_section(self, html_file, possible_section_names, possible_next_sections):

		# Open and parse HTML, then extract all textual content from each paragraph 
		h = html.parse(html_file) #, parser=etree.XMLParser(encoding="utf-8"))
		pars = [paragraph.text_content().lower().encode("UTF-8") for paragraph in h.xpath("//p")]   # .encode("utf-8")

		# First we go backwards trying to find the latest occurrence of 
		# one of the possible names of the section of interest 
		begin = None
		for i in reversed(xrange(len(pars))) :
			if match_any(pars[i], possible_section_names) :
				begin = i
				break

		# If the start wasn't found, just halt right away	
		if (begin is None) :
			return ""

		# Otherwise we can look for the end of the section starting from the start
		# of the found section.
		end = None
		for j in xrange(begin+1, len(pars)) :
			if match_any(pars[j], possible_next_sections) :
				end = j
				break

		# End of section not found, so it's not safe to keep this content, 
		# so we return an empty string.
		if (end is None) :
			return ""

		# Otherwise join all paragraphs inside the section found
		return unicode("".join([fix_hyphens(p) for p in pars[begin:end]]), "UTF-8")


	def get_title_and_abstract(self, paper_id) :
		title, abstract = self.db.select_one(["title", "abstract"], table="papers", where="id='%s'"%paper_id)
		if title is None : title = ""
		if abstract is None : abstract = ""

		return title, abstract


	def process_full_text(self, paper_id):
		'''
		Tokenizes and store in disk the full text of the document provided.
		'''
		txt_file  = config.TXT_PATH % paper_id
		tok_file  = config.TOKENS_PATH % paper_id

		with open(txt_file, 'r') as f :
			text = unicode(f.read(), "utf-8")

		tokens = utils.tokenize(text)
		if (len(tokens) < self.MIN_TOKENS) :
			raise MinimumTokensException('''Minimum number of tokens (%d) could not be extracted. 
			 				Document is likely to be badly encoded.''' % self.MIN_TOKENS)

		self.save_tokens(tokens, tok_file) 


	def process_important_parts(self, paper_id): 
		'''
		Tokenizes some specific parts of the document deemed as important, like
		the title, abstract and conclusion.
		'''
		html_file = config.HTML_PATH % paper_id
		tokens_file = config.TOKENS_PATH_PARTS % paper_id

		# Get title and abstract from DB
		title, abstract = self.get_title_and_abstract(paper_id)

		# Get conclusion from full text
		conclusion = self.get_section(html_file, ['conclusion', 'concluding', 'summary'], 
																	['reference', 'bibliography', 'acknowledg', 'appendix'])

		# Uncomment if you don't want to use the abstract from the DB
#		abstract = self.get_section(html_file, ['abstract'], ['categories', 'keywords', 'introduction'])

		# Tokenize each part and save into a file
		tokens = []
		tokens += utils.tokenize(title)
		tokens += utils.tokenize(abstract)
		tokens += utils.tokenize(conclusion)

		if (len(tokens) < self.MIN_TOKENS) :
			raise MinimumTokensException(("Minimum number of tokens (%d) could not be extracted." % self.MIN_TOKENS) +
											 "Document is likely to have decoding problems." )

		self.save_tokens(tokens, tokens_file)


	def run(self) :

		self.log.info("Starting process %d" % os.getpid())

		# Keep running until a stop file is found
		while (not os.path.exists("stop")) :

			try :
				paper_id = self.tasks.get_next("CONVERTED")

				# Pre-processes the full text and only the important parts to different folders 
				self.process_full_text(paper_id)
				self.process_important_parts(paper_id)

				# Update the task status and the disk in which the file was saved. 
				self.tasks.update_success(paper_id, "TOKENIZED")

				# Everything went OK if got here
				self.log.info("%s: OK" % paper_id)

			# Nothing to collect
			except NothingToProcessException:
				self.log.info("Nothing to process.")
				break

			except MinimumTokensException, e :
				self.log.error("%s: FAIL\n%s\n" % (paper_id, traceback.format_exc()))
				self.tasks.update_error(paper_id, message=str(e))

			# Any other exception we log the traceback and update the DB
			except Exception:
				self.log.error("%s: FAIL\n%s\n" % (paper_id, traceback.format_exc()))
				self.tasks.update_error(paper_id, "TOKENIZE_ERROR")


	def test_extracted_words(self) :
		'''
		Simples manual verification of the extracted tokens.
		'''

		def parse_line(line):
			token, count = line.strip().split()
			return (int(count), unicode(token, "UTF-8"))
	
		ids = self.db.select(fields="id", table="tasks", where="status='TOKENIZED'")
		for id in ids :

			with open(config.TOKENS_PATH_PARTS % id) as f :
				words = map(parse_line, f.readlines())
			
			print "\n-- %s --" % id
			print "\n".join(map(str,sorted(words, reverse=True)[:6]))
				
			call(["google-chrome", "--incognito", "/data/pdf/%s.pdf"%id], stdout=PIPE, stderr=PIPE)



if __name__ == "__main__" :

	nprocs = 1
	if len(sys.argv)>1 :
		nprocs = int(sys.argv[1])

	# If only 1 process do not use multiprocessing for easier debbuging
	if nprocs==1:
		Tokenizer().run()


	zeno.launch(Tokenizer, nprocs)

