'''
Created on Apr 17, 2014

@author: luamct
'''
import os
import traceback
import requests

import sys
import time
from requests.exceptions import ConnectionError
import config
import zeno
from zeno.task_manager import NothingToProcessException
from mymysql.mymysql import MyMySQL
import logging
import utils


class RequestException(Exception):
	''' For handling request errors. '''
	def __init__(self, msg):
		self.msg = msg


class DownloadException(Exception):
	''' For when the requested resource was not found. '''
	pass


class LimitReachedException(Exception):
	''' For when the request limit is reached. '''
	pass


class Downloader() :
	
	def __init__(self):
		'''
		Stores the process id and creates a task manager to get 
		and update tasks.
		'''
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
		self.log = utils.config_logging('downloader', stream=sys.stdout, level=logging.DEBUG,
												format='%(asctime)s (%(name)s) [%(levelname)6s]: %(message)s',
												datefmt="%Y-%m-%d %H:%M:%S")


	def parse_error(self, content):
		'''
		Parsers the returned response's HTML and throws the appropriate exception.
		'''
		if content.find("Download Limit Exceeded"):
			raise LimitReachedException()
		else:
			raise Exception()


	def make_csx_url(self, id):
		return "http://citeseerx.ist.psu.edu/viewdoc/download?doi=%s&rep=rep1&type=pdf" % id

	
	def download_from_csx(self, paper_id):
		''' 
		Downloads the given image URL. 
		''' 

		# Get url from the database
		url = "http://citeseerx.ist.psu.edu/viewdoc/download?doi=%s&rep=rep1&type=pdf" % paper_id
		
		headers = {'User-Agent' : 'Chrome/34.0.1847.116 (X11; Linux x86_64)'}
		response = requests.get(url, headers=headers)

		if (response.status_code != 200) :
			raise RequestException("%d: %s" % (response.status_code, response.reason))

		if response.headers['Content-Type'].startswith('text/html') :
			self.parse_error(response.content)

		# Save file to the local disk
		file_path = os.path.join(self.data_folder, "%s.pdf"%paper_id)
		img_file = open(file_path, "wb")
		img_file.write(response.content)
		img_file.close()
		
	
	def get_all_urls(self, paper_id) :
		''' Returns the external paper URL if available. '''

		cluster_id = self.db.select_one("cluster", table="papers", where="id='%s'" % paper_id)

		alt_paper_ids = self.db.select("id", table="papers", where="cluster=%d" % cluster_id)

		urls = []
		for altern_id in alt_paper_ids :
			urls = urls + [self.make_csx_url(altern_id)]

			other_urls = self.db.select("url", table="urls", where="paperid='%s'" % altern_id)
			urls = other_urls + urls

		return urls


	def download(self, paper_id):
		''' 
		Downloads the given image URL. 
		'''
		headers = {'User-Agent' : 'Chrome/34.0.1847.116 (X11; Linux x86_64)'}

		# Get url from the database
		urls = self.get_all_urls(paper_id)
		for url in urls :

			# Only supports PDF for now
			if url[-3:].lower() != "pdf" :
				continue

			try :
				response = requests.get(url, headers=headers)
			except ConnectionError:
				self.log.warn("Connection error! Ignoring URL '%s'" % (url))
				continue


			response_type = response.headers['Content-Type']

			if response_type.startswith('text/html') :
				if response.content.find("Download Limit Exceeded") >= 0 :
					raise LimitReachedException()
				else: 
					continue

			if (response.status_code != 200) or (response_type != "application/pdf"): 
				continue

# 				raise MissingURLException()
# 			if (response.status_code != 200) :
# 				raise RequestException("%d: %s" % (response.status_code, response.reason))

			# Save file to the local disk
			file_path = config.PDF_PATH % paper_id
			img_file = open(file_path, "wb")
			img_file.write(response.content)
			img_file.close()

			# Download successfully completed			
			return True

		# If we got here, no valid URL was found
		return False


	def run(self) :

		self.log.info("Starting %s." % os.getpid())

		# Keep running until a stop file is found
		while (not os.path.exists("stop")) :

			try :
				paper_id = self.tasks.get_next("START")

				if not self.download(paper_id) :
					raise DownloadException("Could not download paper '%s'." % paper_id)

				# Update the task status and the disk in which the file was saved. 
				self.tasks.update_success(paper_id, "DOWNLOADED")

				# Everything went OK if got here
				self.log.info("%s: OK" % paper_id)

			# Nothing to collect
			except NothingToProcessException:
				self.log.error("Nothing to process.")
				break

			except LimitReachedException:
				self.log.error("Request limit reached!! Waiting...")
				self.tasks.update_release(paper_id, "Request limit reached. Will try again later.")
				time.sleep(60*60)

			# URL missing in the DB or not returning the resource.
			except DownloadException, e:
				self.log.error("%s: FAIL" % (paper_id))
				self.tasks.update_error(paper_id, message=str(e))

			# Request errors
# 			except RequestException, e:
# 				self.log("%s: %s\n%s" % (paper_id, e.msg, traceback.format_exc()), show=True)
# 				self.db.update_status(paper_id, DBManager.DOWNLOAD_ERROR)

			# Any other exception we log the traceback, update the DB and life goes on
			except Exception, e:
				self.log.error("%s: FAIL: %s" % (paper_id, traceback.format_exc()))
				self.tasks.update_error(paper_id, message=str(e))

		# Last thing before exiting thread
		self.close()


	def close(self):
		'''Clean up routine'''
		self.tasks.close()
		self.db.close()
#		self.log_file.close()


#def launch(process):
#	''' Target method for launching the processes '''
#	process.run()


if __name__ == '__main__':

	nprocs = 1
	if len(sys.argv)>1 :
		nprocs = int(sys.argv[1])

	# Removes stop file
	if os.path.exists("stop") :
		os.remove("stop")

	# Doesn't use multiprocessing if len(instances)=1 for ease debugging.
	if nprocs == 1:
		Downloader().run()

	else:
		
		zeno.launch(Downloader, nprocs)
#			proc = mp.Process(target=launch, args=(Downloader(tid),))
#			proc.start()

