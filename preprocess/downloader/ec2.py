'''
Created on Apr 20, 2014

@author: luamct
'''

from boto.ec2 import connect_to_region
import time
import os


class EC2Manager(object):
	'''
	Handles the connection and communication to the EC2 service.
	'''

	def __init__(self):
		''' Open a connection with EC2 service. '''

		self.conn = connect_to_region("us-east-1", 
													 				aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
													 				aws_secret_access_key=os.environ['AWS_SECRET_KEY']
													 				)


	def launch_instances(self, ninstances):
		''' Launch properly configured instance at EC2. '''

		self.conn.run_instances('ami-11617a78',
														 min_count=ninstances,
														 max_count=ninstances, 
						 								 key_name='qcri', 
						 								 security_groups=['Proxies'], 
						 								 instance_type='t1.micro')
		

	def launch_instance_and_wait(self):
		''' Launch a properly configured instance at EC2, wait for it to start 
		running and get its id and DNS address. '''

		reserv = self.conn.run_instances('ami-11617a78', 
										 								 key_name='qcri', 
										 								 security_groups=['Proxies'], 
										 								 instance_type='t1.micro')

		# Get the instance id
		instance_id = reserv.instances[0].id
		state = 'pending'

		# Wait for the instance to be up and running by checking the state
		while (state != 'running') :
			time.sleep(5)
			instance = self.conn.get_only_instances([instance_id])[0]
			state = instance.state

		return (instance.id, instance.public_dns_name)


	def terminate_instance(self, instance_id):
		''' Terminate instance at EC2. '''

		self.conn.terminate_instances([instance_id])

	
	def get_running_instances(self):
		''' List all running proxy instances associated with the current connection.'''
		
		is_ready = lambda i: (i.state=="running" and i.image_id=="ami-11617a78")

		instances = self.conn.get_only_instances()
		return filter(is_ready, instances)


if __name__ == "__main__" :
	
	ec2 = EC2Manager()
	ec2.launch_instances(1)

