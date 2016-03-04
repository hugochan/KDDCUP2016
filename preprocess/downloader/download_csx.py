'''
Created on Apr 17, 2014

@author: luamct
'''
import os
import traceback
import requests

from ec2 import EC2Manager
import zeno
import config
from mymysql.mymysql import MyMySQL
import sys
import logging
import utils
from zeno.task_manager import NothingToProcessException


class RequestException(Exception):
	''' For handling request errors. '''
	def __init__(self, msg):
		self.msg = msg


class NotFoundException(Exception):
	''' For when the requested resource was not found. '''
	pass


class LimitReachedException(Exception):
	''' For when the request limit is reached. '''
	pass


class Downloader() :
	
	def __init__(self, ec2_manager, ec2_instance_id, ec2_instance_dns):
		'''
		Since this is run on the main process, it shouldn't
		open connection or file descriptors.
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

		# EC2 manager to issue commands.
		self.ec2_manager = ec2_manager 

		# EC2 instance information to be used as a proxy.
		self.ec2_instance_id  = ec2_instance_id
		self.ec2_instance_dns = ec2_instance_dns

		# Logging configuration
		self.log = utils.config_logging('downloader', stream=sys.stdout, level=logging.DEBUG,
												format='%(asctime)s (%(name)s) [%(levelname)6s]: %(message)s',
												datefmt="%Y-%m-%d %H:%M:%S")



	def parse_error(self, content):
		'''
		Parsers the returned response's HTML and throws the appropriate exception.
		'''
		if content.find("Document Not Found")>0 :
			raise NotFoundException()
		elif content.find("Download Limit Exceeded"):
			raise LimitReachedException()
		else:
			raise Exception()

# 		html = etree.fromstring(content, etree.XMLParser(resolve_entities=False))
# 		e = html.xpath("//div[@class='error']")
	
		
	def make_url(self, paper_id):
		return "http://%s/viewdoc/download?doi=%s&rep=rep1&type=pdf" % (self.ec2_instance_dns, paper_id)
	

	def download(self, paper_id):
		''' 
		Downloads the given image URL. 
		''' 
		url = self.make_url(paper_id)

		headers = {'User-Agent' : 'Chrome/34.0.1847.116 (X11; Linux x86_64)'}
		response = requests.get(url, headers=headers)

		if (response.status_code != 200) :
			raise RequestException("%d: %s" % (response.status_code, response.reason))

		if response.headers['Content-Type'].startswith('text/html') :
			self.parse_error(response.content)

		# Save file to the local disk
		file_path = config.PDF_PATH % paper_id
		img_file = open(file_path, "wb")
		img_file.write(response.content)
		img_file.close()


	def run(self):

		print "Starting %d." % os.getpid()

		# Keep running until a stop file is found
		while (not os.path.exists("stop")) :

			paper_id = None
			try :
				paper_id = self.tasks.get_next("START")

				self.download(paper_id)

				# Update the task status and the disk in which the file was saved.
				self.tasks.update_success(paper_id, "DOWNLOADED")

				# Everything went OK if got here
				self.log.info("%s: OK" % paper_id)

			# IP limit reached. Terminate the current proxy instance and start another one LOL
			except LimitReachedException, e:
				self.log.error("%s: Limit Reached" % (paper_id))
				
				# Status go back to available, since the problem was not with the paper itself
				self.tasks.update_release(paper_id, "AVAILABLE")

				# Terminate the instance
				self.ec2_manager.terminate_instance(self.ec2_instance_id)

				# Start a new up and wait for it to be ready, storing the id and public DNS.
				self.ec2_instance_id, self.ec2_instance_dns =  self.ec2_manager.launch_instance_and_wait()


			# Nothing to collect
			except NothingToProcessException:
				self.log.error("Nothing to process.")
				break

			# Resource not found errors
			except NotFoundException, e:
				self.log.error("%s: Not Found" % (paper_id))
				self.db.update_error(paper_id, message="Document not found.")

			# Request errors
			except RequestException, e:
				self.log.error("%s: %s\n%s" % (paper_id, e.msg, traceback.format_exc()))
				self.db.update_error(paper_id, message="Error downloading document.")

			# Any other exception we log the traceback, update the DB and life goes on
			except Exception, e:
				self.log.error("%s: FAIL\n%s" % (paper_id, traceback.format_exc()))
				self.db.update_error(paper_id, message=str(e))

		# Last thing before exiting thread
		self.close()


	def close(self):
		'''Clean up routine'''
		self.tasks.close()
		self.db.close()


def launch(process):
	''' Target method for launching the processes '''
	process.run()


if __name__ == '__main__':

	# Removes stop file
	if os.path.exists("stop") :
		os.remove("stop")

	ec2 = EC2Manager()

	instances = ec2.get_running_instances()
	nprocs = len(instances)

	# Doesn't use multiprocessing if len(instances)=1 for ease debugging.
	if len(instances) == 1:
		Downloader(ec2, instances[0].id, instances[0].public_dns_name).run()

	else:
		args = [{'ec2_manager': ec2, 
						 'ec2_instance_id': i.id,
						 'ec2_instance_dns': i.public_dns_name} for i in instances]

		# If args is a list or tuple, then each process gets its own argument dict
		zeno.launch(Downloader, nprocs, args)
	
