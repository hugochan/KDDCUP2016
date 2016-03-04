'''
Created on Apr 12, 2015

@author: luamct
'''
import os
import numpy as np
from scipy.stats.stats import ttest_rel
from scipy.stats import distributions
import config
import cPickle


def paired_ttest(x, y):
	d = x-y
	m = np.mean(d)
	v = np.var(d, ddof=1)

	n = len(x)
	df = n-1
	
	t = m/np.sqrt(v/n)
	p = distributions.t.sf(np.abs(t), df) * 2
	return t, p


def stats_signif(folder, methods1, methods2) :
	'''
	Cross checks statistical significance between all method in `methods1` with
	`methods2` using a t-test. Results are read from `folder`.
	'''
	
	results = {}
	for file_name in os.listdir(folder) :
		file_path = os.path.join(folder, file_name)
		metrics = cPickle.load(open(file_path, 'r'))
		values = np.array(metrics["MAP"])

		results[file_name[:-2]] = values, np.mean(values), np.std(values)

	# Print header with the methods names and 
	# second header with MAP values for reference
	print "\t".join(["Methods", ""] + methods2)
	print "\t".join(["", "MAP@20"] + [u"%.3f \xb1 %.3f" % (results[m][1], results[m][2]) for m in methods2])

#	ttests = np.empty((len(methods1), len(methods2)))
	for _i, m1 in enumerate(methods1) :
		values1, mean1, std1 = results[m1]
		print u"%s\t%.3f \xb1 %.3f\t" % (m1, mean1, std1),
		for _j, m2 in enumerate(methods2) :
			if (m1 != m2) :
	#			ttests[i,j] = ttest_rel(results[m1], results[m2])
				print "%e\t" % paired_ttest(values1, results[m2][0])[1],

			else:
				# Empty cell
				print "-\t",
		print 



if __name__ == '__main__':

	dataset = "csx_dm"
	queryset = "testing"

	stats_signif("%sresults/%s/%s" % (config.DATA, dataset, queryset),
								["MultiLayered", "TopCited(G)", "PageRank(G)"], 
								["MultiLayered", "TopCited(G)", "PageRank(G)", "BM25", "TF-IDF", "TopCited", "PageRank(pre)", "PageRank(pos)"])

	# Max 2 layers 
#	stats_signif(config.DATA + "results/layers", 
#								["P", "PA", "PT", "PW", "PV"], 
#								["P", "PA", "PT", "PW", "PV"])

	# All combinations
#	all = ["P", "PA", "PT", "PW", "PV",
#				 "PAT", "PAW", "PAV", "PTW", "PTV", "PWV",
#				 "PTWV", "PAWV", "PATV", "PATW", "PATWV"]
#	stats_signif(config.DATA + "results/layers", all, all)

	# Incremental
#	atts = ['A', 'C', 'Q', 'AC', 'CQ', 'AQ', 'ACQ']
#	stats_signif(config.DATA + "results/atts", atts, atts)

#	methods = ["none", "ngrams", "extracted", "extended", "both"]
#	stats_signif(config.DATA + "results/kws", methods, methods)
	

