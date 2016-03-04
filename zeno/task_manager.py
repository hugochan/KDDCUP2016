'''
Created on Apr 17, 2014

@author: luamct
'''

import sys
import MySQLdb


ZENO_DB_NAME = "zeno"


class NothingToProcessException(Exception):
	""" Raised when there are no more available pins on the DB """
	pass


class TasksManager:


	def __init__(self, group, host="localhost", user="root", passwd="") :

		self.db = MySQLdb.connect(host=host, 
														user=user, 
														passwd=passwd, 
														db=ZENO_DB_NAME,
														charset="utf8",
														unix_socket="/var/run/mysqld/mysqld.sock",
														init_command="SET AUTOCOMMIT=0")

		# Table used for controlling the access
		self.table = group


	def get_next(self, state):
		'''
		Returns one available task.
		'''
		cursor = self.db.cursor()

		# Query for next available user using FOR UPDATE to avoid race conditions
		query = """SELECT id FROM %s
						 	 WHERE state='%s' AND status='AVAILABLE' 
						 	 ORDER BY id DESC LIMIT 1 
						 	 FOR UPDATE""" % (self.table, state)
		cursor.execute(query)

		result = cursor.fetchone()
		if (result == None) :
			raise NothingToProcessException()

		# Drop tuple and unicode type 
		task_id = str(result[0])

		# Mark the account as being processed
		query = """UPDATE %s
							 SET status='IN_PROGRESS' 
							 WHERE id='%s'""" % (self.table, task_id)
		cursor.execute(query)

		self.db.commit()
		cursor.close()

		return task_id

	
	def update_task(self, task_id, state=None, status=None, message=None) :
		''' Update task execution status. '''

		sets = {}
		if state!=None: sets['state'] = state
		if status!=None: sets['status'] = status
		if message!=None: sets['message'] = message

		set_clause = ", ".join(['%s="%s"' % (name,value) for name, value in sets.items()])

		cursor = self.db.cursor()
		query = "UPDATE %s SET %s WHERE id='%s'" % (self.table, set_clause, task_id)
		cursor.execute(query)
		self.db.commit()
		cursor.close()


	def update_success(self, task_id, state) :
		''' Updates task status as successful and ready for next state. '''

		self.update_task(task_id, state, 'AVAILABLE')


	def update_error(self, task_id, message) :
		''' Updates task status as error and adds a small message for debugging.'''

		self.update_task(task_id, status='FAILED', message=message)


	def update_release(self, task_id, message=None) :
		''' 
		Releases the task to be processed by another process, probably due 
		to same error in this process. Use only when the error is not likely to occur 
		again, otherwise this task will continuously attempted and failed.
		'''
		self.update_task(task_id, status='AVAILABLE', message=message)


	def close(self) :
		self.db.close()
