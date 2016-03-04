'''
Created on Aug 18, 2015

@author: luamct
'''
import zeno
import time
import os
import random


class MyProcessor :

	def __init__(self, name, age):
		'''
		Constructor
		'''
		self.tasks = zeno.TasksManager("tasks")
	

	def run(self) :

		while not os.path.exists("stop"): 
		
			task = self.tasks.get_next("START")
			print "Processing task '%s'..." % task
			time.sleep(2)
			
			if random.random()>0.2 :
				print "Processed!"
				self.tasks.update_sucess(task, "DOWNLOADED")
			else: 
				print "Failed"
				self.tasks.update_error(task, "Problem downloading PDF.")


if __name__=="__main__" :
	
	# Removes stop file
	if os.path.exists("stop") :
		os.remove("stop")
	
	zeno.launch(MyProcessor, nprocs=2, args={'name': 'Luam', 'age':29})


