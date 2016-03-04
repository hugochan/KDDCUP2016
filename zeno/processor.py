'''
Created on Jun 3, 2014

@author: luamct
'''

'''
Created on Apr 18, 2014

@author: luamct
'''
#from db import DBManager
import multiprocessing as mp

import os
import time



class ProcessorBase():

	def __init__(self) :
		''' 
		Open the DB connection. This method should run in the forked process, 
		otherwise the DB connection will fail.
		'''
#		self._db = DBManager(host=self.config["host"], 
#												user=self.config["user"],
#												passwd=self.config["passwd"], 
#												db_name=self.config["db_name"])
		pass

	def init_process(self):
		''' 
		Processor base class. Must be inherited by every processing class. 
		'''
#		super(ProcessorBase,self).__init__()
		pass


	def init(self) :
		''' 
		User's pseudo constructor. Should be overwritten if user needs it to 
		be called before the process method. The main distinction from the
		__init__ native constructor is that this method is called in the new 
		process, so file handlers, DB and HTTP connections will be within scope
		of the process call.
		'''
		pass


#	def _load_config(self) :
#		'''
#		Loads configuration file. It may be dropped in future versions.
#		'''
#		config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config") 
#		self.config = dict(l.strip().split('=') for l in open(config_path))
#
#		
#	def _init_db(self):
#		''' 
#		Open the DB connection. This method should run in the forked process, 
#		otherwise the DB connection will fail.
#		'''
#		self._db = DBManager(host=self.config["host"], 
#												user=self.config["user"],
#												passwd=self.config["passwd"], 
#												db_name=self.config["db"])


#	def _init_log(self):
#		''' 
#		Open the log files. This method should run in the forked process, 
#		otherwise the file handlers will fail.
#		'''
#		_log_folder = "log"
#		if (not os.path.exists(_log_folder)) :
#			os.makedirs(_log_folder)
#
#		self._log_file = open("%s/%s.log" % (_log_folder, socket.gethostname()), "a")

	
	def get_next(self, from_state):
		return self._db.get_next(from_state)
		

	def update_status(self, task_id, to_status):
		self._db.update_status(task_id, to_status)


	def log(self, message, show=False) :
		'''
		File for multi-threading logging.
		'''
		# If there's something to log
		if message:
			print >> self._log_file, message,
			self._log_file.flush()
			if show:
				print message


	def start(self):
		'''
		Entry point for the subprocess. Note that initialization calls are postponed until this
		point so they run already in the new process. Otherwise, file handlers and DB connections
		would likely fail.
		'''

		# Open DB connection and the log files.
		self._load_config()
		self._init_db()
		self._init_log()

		# User pseudo constructor
		self.init()

		# Call main processing method. Must be implemented by child class.
		self.process()

		# Last thing before exiting thread
		self._close()


	def process(self):
		print "Child class must overwrite this method."
		time.sleep(3)


	def _close(self):
		'''Clean up routine'''
		self._db.close()
		self._log_file.close()


def launcher(cls, args) :
	''' 
	Actual entry point for the new process. Constructs the processor
	object with the given class and arguments, then starts processing.
	'''

	proc = cls(**args)
	proc.run()


def launch(cls, nprocs=1, args={}) :
	'''
	Launches <nprocs> instances of class <cls> using the given arguments.
	'args' should be a dictionary with the names and values as parameters
	for the constructing class 'cls'. 
	
	If args is a list or tuple, then each process gets it's corresponding 
	args from the list, which must be of size nprocs.
	'''

	# Removes stop file
	if os.path.exists("stop") :
		os.remove("stop")

	for i in xrange(nprocs) :
		
		# If args is a list, then each process gets it's 
		# corresponding args from the list. 
		if isinstance(args, (list, tuple)) :
			proc_args = args[i]
		else: 
			proc_args = args

		proc = mp.Process(target=launcher, args=(cls, proc_args))
		proc.start()

