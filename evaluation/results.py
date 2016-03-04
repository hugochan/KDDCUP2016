'''
Created on Jul 21, 2015

@author: luamct
'''

import config
import numpy as np
import cPickle
import os
from collections import defaultdict
import metrics
from evaluation.metrics import apk, ndcg


LATEX_TEMPLATE = """
\\documentclass{/var/tmp/tex/vldb}
\\usepackage{array}
\\newcolumntype{L}[1]{>{\raggedright\let\newline\\\arraybackslash\hspace{0pt}}m{#1}}
\\newcolumntype{R}[1]{>{\raggedleft\let\newline\\\arraybackslash\hspace{0pt}}m{#1}}
\\newcolumntype{C}[1]{>{\centering}m{#1}}

\\begin{document}

\\begin{table*}[!t]
\\scriptsize
\\centering
\\begin{tabular}{|c||c|c||c|c||c|c||c|c||}
\\hline
{\it Datasets} & \\multicolumn{2}{|c||}{\\bf CSX} & 
\\multicolumn{2}{|c||}{\\bf AMiner} &
\\multicolumn{2}{|c||}{\\bf CSX ML} &
\\multicolumn{2}{|c||}{\\bf CSX DM} \\\\
\\hline
{\it Method} & MAP@20 & NDCG@20 & MAP@20 & NDCG@20 & MAP@20 & NDCG@20 & MAP@20 & NDCG@20 \\\\
\\hline
%s
\\end{tabular}
\\caption{Performance Comparison for Manual, Surveys, and Citations Query Sets}
\\vspace{-0.1in}
\\label{tab:results}
\\end{table*}
\end{document}"""


def get_values(method, dataset, query_set, metrics) :

  file_path = "%s/results/%s/%s/%s.p" % (config.DATA, dataset, query_set, method)
  if not os.path.exists(file_path) :
#		return ["-"]*len(metrics)
    return [None]*2*len(metrics)

  MAP = []
  NDCG = []
  results = cPickle.load(open(file_path, 'r'))
  for correct, relevances, returned in results :
    MAP.append(apk(correct, returned, 20))
    NDCG.append(ndcg(correct, returned, relevances, 20))


  values = [np.mean(MAP), np.std(MAP),
            np.mean(NDCG), np.std(NDCG)]

  return values


def find_best(subtable, columns) :
  subtable = np.array(subtable)
  return {col: np.argmax(subtable[:,col]) for col in columns}


def to_latex_table() :

  query_sets = ["manual", "surveys", "testing"]

  datasets = ["csx", "aminer", "csx_ml", "csx_dm"]

  methods = [("MultiLayered", "Multi-Layered"),
             ("PageRank(G)", "PageRank ($G_q$)"),
             ("TopCited(G)", "TopCited ($G_q$)"),
             ("BM25", "Okapi BM25"),
             ("TF-IDF", "TF-IDF"),
             ("TopCited", "TopCited"),
             ("CiteRank", "CiteRank"),
             ("PageRank(pre)", "PageRank (pre)"),
             ("PageRank(pos)", "PageRank (pos)"),
             ("GoogleScholar", "GoogleScholar"),
             ("ArnetMiner", "ArnetMiner"),
             ("Meng", "Meng")]

  table = []
  for query_set in query_sets :
    subtable = []
#		rows.append(header)

    for i, (method, tex_name) in enumerate(methods) :

      # Start the row with the method's name
      subtable.append([tex_name])

      for j, dataset in enumerate(datasets) :
        values = get_values(method, dataset, query_set, ["MAP", "NDCG@20"])
        subtable[i].extend(values)

    table.append((query_set, subtable))
    print ""

  # To place a line below row 3
  line_at = 2

  rows = []
  for query_set, subtable in table :

    header = "\\multicolumn{9}{|c||}{\\bf %s Query Set}\\\\\n\\hline\n" % (query_set.capitalize())
    rows.append(header)

    best = find_best(subtable, range(1, len(subtable[0]), 2))

    for j, row in enumerate(subtable) :
      row_str = [row[0]]

      for i in xrange(1, len(row), 2) :
        if (row[i] is not None) :
          row_str.append(u"%s{%.3f $\\pm$ %.3f}" % ('\\textbf' if best[i]==j else '', row[i], row[i+1]))
        else :
          row_str.append("-")

#			line_break = " \\\\ \\hline\n"
      # Add the method's name, join the cells with & and add a line break
      row_str = " & ".join(row_str) + " \\\\\n"
      rows.append(row_str)
      if j==line_at:
        rows.append("\\hline\n")

    rows.append("\\hline\n")

# Join all rows	
  data = "".join(rows)

  with open("/var/tmp/tex/results.tex", 'w') as file :
    print >> file, LATEX_TEMPLATE % data

  os.chdir("/var/tmp/tex")
  os.system("pdflatex results.tex")
  # os.system("cp results.pdf results.tex /home/luamct/ca/latex/CitationExplorer/vldb/")


if __name__ == '__main__':
  to_latex_table()

