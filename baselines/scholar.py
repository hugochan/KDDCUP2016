'''
Created on Jul 22, 2015

@author: luamct
'''

from evaluation.query_sets import load_query_set
import config
import os
import time


def save_scholar_results(query_set, n):
	'''
	For each given query, request on google scholar, search for textually similar
	entries on the index and show them to the user for confirmation.
	'''
	from scholar_api import ScholarQuerier, SearchScholarQuery

	queries = load_query_set(query_set)

	querier = ScholarQuerier()
	scholar_query = SearchScholarQuery()

	# Folder to store saved results
	# from_folder = config.QUERY_SETS_PATH + "manual"
	save_folder = "%s/scholar/%s" % (config.DATA, config.DATASET)

	for query, query_id, _, _, _, _ in queries:

		# Only searches if file doesn't already exists
		file_path = "%s/%s.txt" % (save_folder, query_id)
		if os.path.exists(file_path):
			continue

		time.sleep(1)

		print "\nProcessing '%s'" % query,
		scholar_query.set_words(query)

		# We stop requesting new pages once we found at least n
		start = 0
		titles = []
		while (len(titles) < n):

			scholar_query.set_start(start)
			querier.send_query(scholar_query)

			# Check if we got a captcha as response
			if (len(querier.articles) == 0):
				if (start == 0):
					raise Exception("Request probably got blocked due to overload.")
				else:
					# If we got 0 article after some requests it may be that
					# all articles were fetched. Skip to next query.
					break

			# Get only titles and try to find entries in our dataset for them
			for article in querier.articles:
				titles.append(article['title'].strip('. '))

			# Set correct pagination
			start += 20


		# Write to file
		with open(file_path, 'a') as file:
			print >> file, query
			for title in titles:
				print >> file, "%s" % (title.encode("UTF-8"))


def match_by_title(query, result_titles, correct_titles):
	import jaro

	# print '\n%s' % query
	# print '\n'.join(result_titles), '\n'
	# print '\n'.join(sorted(correct_titles)), '\n'

	# Apply lower case in all entries first
	correct_titles = map(unicode.lower, correct_titles)

	matched_ids = []
	for rtitle in result_titles:

		rtitle = rtitle.lower()

		best_jw = 0.0
		best_id = 0
		for i, ctitle in enumerate(correct_titles):
			jw = jaro.jaro_winkler_metric(unicode(rtitle), unicode(ctitle))
			if (jw > best_jw):
				best_jw = jw
				best_id = i

		# Check if we have a match
		if (best_jw >= 0.9):
			matched_ids.append(str(best_id))
		else:
			matched_ids.append('')

	# Create an incremental list of string ids for the correct
	# titles, since they are not in the database
	correct_ids = map(str, range(len(correct_titles)))

	return correct_ids, matched_ids

# for i, rtitle in enumerate(results) :
#		print rtitle
#		print titles[int(matched_ids[i])] if (matched_ids[i]) else "No match!"
#		print

#		print rtitle
#		print titles[best_id]
#		print "Best match (%.3f) is %d\n" % (best_jw, best_id)

#	return queries_cits



if __name__ == '__main__':
	#	import jaro
	#	print jaro.jaro_winkler_metric(u"inverting schema mappings", u"inverting schema mappings")

	#	query = "data exchange"
	#	gs = GoogleScholarSearcher()
	#	r = gs.search(query, limit=10)
	#	match_by_title(query, r, "manual")
	#	print "\n".join(results)

	save_scholar_results("surveys", n=20)


