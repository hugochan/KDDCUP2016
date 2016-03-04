'''
Created on May 28, 2014

@author: luamct
'''
import nltk.data
from lxml import html, etree
import re
import numpy as np

nltk.data.path.append('/home/luamct/.nltk_data/')
tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")


# POSITIVE PATTERNS
THIS_WORK   = re.compile("this\s+(paper|work|article|chapter|tutorial)")
WE          = re.compile('(\s|^)+we\s+')
CONTRIBS    = re.compile('contribution(s)?')
RESULTS     = re.compile('\s+our\s+(result(s)+|methodology|framework|architecture)')
VERBS       = re.compile('show|prove|describe|give|provide|present|demonstrate|' + 
												 'explore|implement|derive|introduce|investigate|extend|' + 
												 'address|conclude|evaluate|propose|develop|study')

WE_VERBS    = re.compile('(\s|^)+we\s+(.*)(show|prove|describe|give|provide|present|demonstrate|' + 
												 'explore|implement|derive|introduce|investigate|extend|' + 
												 'address|conclude|evaluate|propose|develop|study|focus)')

# NEGATIVE PATTERNS
LICENSE = re.compile("license|copy")
SECTION = re.compile("section")
ACKS = re.compile("supported|funded|grant|thank")


regexs = {
				THIS_WORK : 0.5,
				WE : 0.2,
# 				VERBS : 0.4,
				CONTRIBS : 1.0,
				RESULTS : 0.5,
				WE_VERBS : 1.0,
				LICENSE : -1.0,
				SECTION : -0.5,
				ACKS : -1.5
				}


class SectionNotFound(Exception) :
	pass


def max_subarray(a, limit):

	max_here = a[0]
	begin = end = 0
	
	results = []
	for i, n in enumerate(a) :
		if max_here<0 :
			max_here = n
			begin = i
		else: 
			max_here += n
			
# 		if max_here >= max_all :
		if max_here >= limit :
# 			max_all = max_here

			end = i
			results.append((begin,end))

	return results
	

def calc_score(sentence) :
	
	s = sentence.lower().strip()

	# Stores the aggregated score for the sentence
	score = 0.0

	for regex, score_inc in regexs.items() :
		
		if re.search(regex, s) :
# 			print regex.pattern
			score += score_inc

	if not score :
		score = -0.5
		
	return score



def positive_ones(nsentences, results) :
	pos = np.zeros(nsentences).astype(np.bool)
	for begin, end in results :
		pos[begin:end+1] = True

	return pos
	

def mark_contribs(html_file, marked_html_file) :

	h = html.parse(html_file)
# 	text = "".join([ p.text_content() for p in h.xpath("//p") ])

	pars = h.xpath("//p")
	
	for par in pars :

		# Get the paragraph's text fixing the hyphenation
		text = par.text_content().replace("-\n", "")
		
		sentences = tokenizer.tokenize(text.strip())
		scores = map(calc_score, sentences)
	
		intervals = max_subarray(scores, 1.0)
		mask = positive_ones(len(sentences), intervals)

		par.clear()

		texts = []
# 		text = ''
# 		marked_sentences = []
		for i, s in enumerate(sentences) :
			if mask[i] :
				marked = etree.Element("font", style="background-color:yellow", score=str(scores[i]))
				marked.text = s
				marked.tail = ''
				par.append(marked)

			else :
				if len(par):
					marked = par[-1]
					marked.tail += ' ' + s
				else: 
					texts.append(s)

		
		par.text = ' '.join(texts)

	h.write(marked_html_file, pretty_print=True, method="html")


def get_section(html_file, section_name, possible_next_sections):
	
	h = html.parse(html_file)
	pars = h.xpath("//p")
	
	begin = end = -1
	for i, par in enumerate(pars) :
		
		if (begin>0) and (end>0) :
			break

		par_text = par.text_content().lower()
		if begin<0 and (par_text.find(section_name, 0, 20) >= 0) :
			begin = i

		if begin>=0 :
			for next_section in possible_next_sections :
				if (par_text.find(next_section, 0, 20) >= 0) :
					end = i

	text = ""
	if (begin<0) or (end<0) :
		raise SectionNotFound("Section %s not found."%section_name)

		text = "".join([par.text_content() for par in pars[begin:end]])

	return text


def fix_hyphens(text):
	return text.replace("-\n", "")


if __name__ == '__main__':
	
	htmlfile = '/data/htmls/10.1.1.336.9303.html'
	htmlfile = '/data/htmls/10.1.1.295.7727.html'
	htmlfile = '/data/htmls/10.1.1.19.2116.html'

# 	print fix_hyphens(get_section(htmlfile, 'abstract', ['introduction']))
	print fix_hyphens(get_section(htmlfile, 'conclusion', ['references', 'bibliography', 'acknowledges', 'appendix']))
	

# 	s = "We have supported"
# 	print re.search(ACKS, s.lower())

# 	h = html.fromstring('''<html> <body> <p>before <b>bold</b> after <i>italic</i></p></body> <html>''')
# 	p = h.xpath("//p")[0]
# 	print [t for t in p.itertext()]
# 	print p.getchildren()[0].tail
	
# 	s = "Our contributions are: 1. we have proved that..."
# 	s = 'In\naddition, K. Passino and K. Burgess were also supported in part by National Science Foundation Grant IRI-9210332.'
# 	s = "CONCLUSION\nWe have proposed a new framework, referred to as DAM,\nfor multiple source domain adaptation."
# 	print calc_score(s)

# 	mark_contribs('/data/htmls/10.1.1.63.3943.html', '/tmp/marked.html')

# 	show('/data/htmls/10.1.1.261.3172.html')
# 	show('/data/htmls/10.1.1.327.1631.html')
# 	show('/data/htmls/10.1.1.89.5012.html')
# 	show('/data/htmls/10.1.1.5.7908.html')

# 	a = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
# 	print max_subarray(a, 5)

# 	print tokenizer.tokenize("Contributions: 1. whatevis; 2. whatevis! again.")
	