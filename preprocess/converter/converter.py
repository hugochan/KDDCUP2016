'''
Created on Apr 18, 2014

@author: luamct
'''
#from dbmanager import DBManager, NothingToProcessException
from lxml import html
import config

import os
import traceback
import sys
import time
import subprocess
import zeno
from mymysql.mymysql import MyMySQL
import utils
import logging
from zeno.task_manager import NothingToProcessException



class Converter:

	def __init__(self):
		''' Converter constructor.'''

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
		self.log = utils.config_logging('converter', stream=sys.stdout, level=logging.DEBUG,
												format='%(asctime)s (%(name)s) [%(levelname)6s]: %(message)s',
												datefmt="%Y-%m-%d %H:%M:%S")


	def convert_pdfbox(self, infile, outfile) :
		'''
		Convert PDF to HTML using a command line tool. 
		'''
		proc = subprocess.Popen(["java", "-jar", config.PDFBOX_PATH, "ExtractText", "-html", "-force", infile, outfile], stderr=subprocess.PIPE)

		# Log any output from the command
		for line in iter(proc.stderr.readline, '') :
			self.log.debug(line.strip())

		code = proc.wait()
		if code!=0 :
			raise Exception("Could not convert file %s to HTML with pdfbox." % infile)


	def tohtml(self, paperid):
		'''
		Convert PDf to HTMl using PDFBox command line tool.
		'''
		infile  = config.PDF_PATH % paperid
		outfile = config.HTML_PATH % paperid

		self.convert_pdfbox(infile, outfile)


	def totxt(self, paperid):
		'''
		Converts HTML to pure text by extracting all text elements from the the HTML.  
		'''
		infile  = config.HTML_PATH % paperid
		outfile = config.TXT_PATH % paperid

		h = html.parse(infile)
		pars = h.xpath("//p")
		text = ''.join([par.text_content() for par in pars])
		text = text.replace("-\n", "")
	
		with open(outfile, 'w') as f :
			f.write(text.encode("UTF-8"))


	def run(self):

		print "Starting %s." % os.getpid()

		# Keep running until a stop file is found
		while (not os.path.exists("stop")) :

			paper_id = None
			try :
				paper_id = self.tasks.get_next('DOWNLOADED')

				self.tohtml(paper_id)
				self.totxt(paper_id)

				# Update the task status and the disk in which the file was saved. 
				self.tasks.update_success(paper_id, 'CONVERTED')

				# Everything went OK if got here
				self.log.info("%s: OK" % paper_id)

			# Nothing to collect
			except NothingToProcessException:
				self.log.warn("Nothing to process.")
				time.sleep(10)

			# Any other exception we log the traceback, update the DB and life goes on
			except Exception, e:
				self.log.error("%s: FAIL\n%s\n" % (paper_id, traceback.format_exc()))
				self.tasks.update_error(paper_id, message=str(e))

		# Last thing before exiting thread
		self.close()


	def close(self):
		'''Clean up routine'''
		self.db.close()
		self.tasks.close()


if __name__ == '__main__':

	nprocs = 1
	if len(sys.argv)>1 :
		nprocs = int(sys.argv[1])

	# Removes stop file
	if os.path.exists("stop") :
		os.remove("stop")

	# Doesn't use multiprocessing if nprocs=1 to enable debugging.
	if nprocs==1:
		Converter().run()

	else:
		zeno.launch(Converter, nprocs)

