'''
Created on Jun 16, 2014

@author: luamct
'''

import MySQLdb


class MyMySQL():
	'''
	Simple and non-comprehensive (yet handy) wrapper for MySQLdb.
	'''

	def __init__(self, db, host="localhost", user="root", passwd="", **params):
		'''
		Opens database connection.
		'''
		self.db = MySQLdb.connect(host=host,
															db=db,
															user=user,
															passwd=passwd,
															charset="utf8",
															unix_socket="/tmp/mysql.sock",
															**params)

	def create_table(self, table_name, table_description):
		if not isinstance(table_name, str) or not isinstance(table_description, list):
			raise TypeError("table_name should be a string and table_description should be a list of strings")

		query = "CREATE TABLE IF NOT EXISTS %s (" % table_name
		for each in table_description[:-1]:
			query =  query + each + ','
		query = query + table_description[-1] + ');'

		cursor = self.db.cursor(MySQLdb.cursors.Cursor)
		cursor.execute(query)
		# print query
		cursor.close()


	def select_query(self, query) :
		'''
		Simple select query wrapper given that the query is already assembled.
		'''
		cursor = self.db.cursor(MySQLdb.cursors.Cursor)
		cursor.execute(query)
		rows = cursor.fetchall()
		cursor.close()
		return rows


	def select(self, fields, table, join_on=None, where=None, order_by=None, limit=None) :
		'''
		Assembles and executes select queries. If the 'query' parameter is passed all other
		parameters are ignored and the query is executed as provided.
		'''


		# Check if it's a join or a single table
		if isinstance(table, basestring) :
			table_str = table

		elif hasattr(table, '__iter__') :
			if not join_on :
				raise TypeError("Parameter 'join_on' must be provided if 'table' is a list.")

			t1, t2 = table
			f1, f2 = join_on
			table_str = "%s JOIN %s ON %s.%s = %s.%s" % (t1, t2, t1, f1, t2, f2)

		# Check if it's a single field or multiple fields,
		# in which case we merge them properly.
		single_field = False
		if isinstance(fields, basestring) :
			fields_str = fields

			if fields != "*" :
				single_field = True

		elif hasattr(fields, '__iter__'):
			fields_str = ", ".join(fields)

		else :
			raise TypeError("Parameter 'fields' is of unsupported type. String or iterable needed.")

		where_str = ""
		if where :
			where_str = "WHERE %s" % where

		order_by_str = ""
		if order_by :
			if (not isinstance(order_by, basestring)) and len(order_by)==2 :
				order_by_str = "ORDER BY %s %s" % (order_by[0], order_by[1])
			else :
				order_by_str = "ORDER BY %s" % order_by

		limit_str = ""
		if limit :
			limit_str = "LIMIT %d" % limit

		query = "SELECT %s FROM %s %s %s %s" % (fields_str, table_str, where_str, order_by_str, limit_str)

		# Execute query
		rows = self.select_query(query)

		# Check if we return as list of tuples (if multiple fields were requested) or a
		# list of single values (only one field was requested).
		if single_field :
			rows = [row for (row,) in rows]
		else :
			rows = list(rows)

		return rows


	def select_one(self, fields, table, join_on=None, where=None, order_by=None, limit=None) :
		result = self.select(fields, table, join_on, where, order_by, limit)
		return result[0] if len(result)>0 else None


# 	def execute(self, query, load_all=True) :
#
# 		if load_all :
# 			cursor = self.db.cursor(MySQLdb.cursors.Cursor)
# 			cursor.execute(query)
# 			rows = cursor.fetchall()
# 			cursor.close()
# # 			if single_field :
# # 				rows = [row for (row,) in rows]
#
# 			return list(rows)
#
# 		else :
# 			cursor = self.db.cursor(MySQLdb.cursors.SSCursor)
# 			cursor.execute(query)
#
# 			for row in cursor :
# 				yield row
#
# 			cursor.close()


	def assemble_values(self, values):

		quoted = []
		for v in values :
			if v is None :
				quoted.append("NULL")
			elif isinstance(v, basestring) :
				quoted.append("'%s'" % v.replace("'", "").replace("\\", ""))
			else :
				quoted.append("%g" % v)

		return "(%s)" % (",".join(quoted))


	def insert_query(self, query) :
		'''
		Simple method to execute insertion queries as they are provided.
		A commit is always performed.
		'''
		cursor = self.db.cursor()
		cursor.execute(query)
		cursor.close()
		self.db.commit()


	def insert(self, into, fields, values, ignore=False, query=None) :
		'''
		Insert new values into a table.
		'''
		# If 'query' is provided, ignore all other parameters and execute query as it is.
		if not query :

			# Just return if there are no values to insert
			if len(values)==0 :
				return

			# If the first element is iterable (but not a string), then multiple values
			# were passed and should be inserted all at once.
			if not hasattr(values[0], '__iter__') or (isinstance(values[0] , basestring)) :
				values = [(v,) for v in values]

			values = map(self.assemble_values, values)
#			else :
#				values = [self.assemble_values(values)]


			query = "INSERT "
			if ignore:
				query += "IGNORE "

			fields = ",".join(fields)
			values = ",".join(values)
			query += " `%s` (%s) VALUES %s" % (into, fields, values)

		# Execute query
		self.insert_query(query)


	def update(self, table, set, where=None) :
		'''
		Update values given by 'set' in the 'table' fitting the condition 'where'.
		'''

		where_str = ""
		if where:
			where_str = "WHERE %s" % where

		query = "UPDATE %s SET %s %s" % (table, set, where_str)

		cursor = self.db.cursor()
		cursor.execute(query)
		cursor.close()
		self.db.commit()


	def delete(self, table, where) :
		'''
		Update values given by 'set' in the 'table' fitting the condition 'where'.
		'''
		query = "DELETE FROM %s WHERE %s" % (table, where)

		cursor = self.db.cursor()
		cursor.execute(query)
		cursor.close()
		self.db.commit()


	def close(self) :
		self.db.close()



if __name__ == "__main__" :
	pass
	# import config
	# db = MyMySQL(config.DB_NAME, user=config.DB_USER, passwd=config.DB_PASSWD)
	# db.create_table('names',['firstname VARCHAR(10)', 'lastname VARCHAR(10)'])
# 	import chardet
# 	print u'\u0093'.encode('utf-8')

# 	for idx, id in enumerate(db.select("id", table="tasks", where="status='TOKENIZED'")) :
# 		abstract = db.select_one("abstract", table="papers", where="id='%s'"%id)
# 		print abstract
# 		if abstract :
# 			print abstract #.encode("utf-8")
# 		continue

# 		if abstract :
# 			enc = chardet.detect(abstract)["encoding"]
# 			if enc != "ascii" :
# 				print "%15s: %12s\t %s" % (id, enc, unicode(abstract, "ISO-8859-1").encode("UTF-8"))

# 		if idx==100 :
# 			break

# 		print chardet.detect(rows)

# 	db = MyMySQL(db="csx", user="root")
# 	print len(db.select(fields=["citing", "cited"], table="graph"))
# 	print len(db.select(fields="citing", table="graph"))



