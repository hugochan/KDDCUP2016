'''
Created on Oct 24, 2014

Stores global configuration variables. Assuming Unix system due
to path separators.

@author: luamct
'''
# export PYTHONPATH=$PYTHONPATH:/Users/hugo/Documents/Study\&Work/Research/Projects/citation_networks/KDDCUP/software/FrontierGo

# Main data folder and dataset to be used
DATA = '/Users/Hugo/Documents/Study&Work/Research/Projects/citation_networks/KDDCUP/software/data/'
DATASET = 'mag'
#DATASET = 'aminer'
#DATASET = 'csx_ml'
#DATASET = 'csx_dm'

# MySQL config
DB_HOST = 'localhost'
DB_NAME = DATASET
DB_USER = 'fos'
DB_PASSWD = 'passfos'

ZENO_DB_NAME = "zeno"

# Folder locations for some required data
PDF_PATH = DATA + "pdf/%s.pdf"
TXT_PATH = DATA + "txt/%s.txt"
HTML_PATH = DATA + "html/%s.html"

# These are pre-tokenized folders, for efficiency
TOKENS_PATH  = DATA + "tokens/%s.txt"
TOKENS_PATH_PARTS = DATA + "tokens_parts/%s.txt"

# External tool used to convert PDFs to text and html
PDFBOX_PATH = "/home/luamct/apps/pdfbox/pdfbox.jar"

INDEX_PATH = DATA + "index_" + DATASET
CACHE_FOLDER = DATA + "cache/"
CTXS_VOCAB_PATH = DATA + "contexts_tfidfs_tokens.txt"
CTX_PATH = DATA + "contexts_tfidfs/%s.txt"

#TOKENS_PATH = TOKENS_PATH_PARTS

# Evaluation and ground truth settings
MENG_GRAPH_PATH = DATA + "baselines/%s_meng.gexf" % DATASET
CITERANK_FILE_PATH = DATA + "baselines/citerank/%s.p" % DATASET

# Placeholders are for the dataset and query set, respectively (e.g.: 'csx' and 'citations')
QUERY_SETS_PATH = DATA + "query_sets/" + DATASET + "/"

# Won't have any effect, for now (topics are pre-processed in the DB)
NTOPICS = 300

MIN_NGRAM_TFIDF = 0.2
MIN_TOPIC_VALUE = 0.1

# What kind of keywords to include. Supports:
#  + ngrams : Standard top TF-IDF ngrams.
#  + extracted: Only keywords extracted directly from the text (low recall)
#  + extended: Only keywords derived from frequency and based in the
#                   extracted keywords.
#  + both: Both of the above. Still applies the MIN_NGRAM_TFIDF threshold though,
#          however extracted keywords have value 1.0, so they'll always get in.

KEYWORDS = "ngrams"

# The placeholders are for parameters <dataset>, <K> and <H>, correspondingly
IN_MODELS_FOLDER = DATA + "models/%s_%d_%d"
OUT_MODELS_FOLDER = "out_models"


PARAMS = {
	'K': 10,
	'H': 1,
	'min_topic_lift': 1.0,
	'min_ngram_lift': 1.0,
	'papers_relev': 0.25,
	'authors_relev': 0.25,
	'words_relev': 0.25,
	'topics_relev' : 0.0,
	'venues_relev': 0.25,
	'alpha': 0.3,
	'query_relev': 5.0,
	'age_relev': 0.005,
	'ctx_relev': 0.5
}
