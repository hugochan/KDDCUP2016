'''
Created on Apr 27, 2014

@author: luamct
'''

import re
import os
from collections import defaultdict
import string


class ReferencesNotFound(Exception) :
	pass

class NoCitationsFound(Exception):
	pass


def files_in(folder):
	''' Generator for the path of all files in the given folder. '''
	for txtfile in os.listdir(folder) :
		yield os.path.join(folder, txtfile)


def clean(s):
	''' Simple line break removal.'''
	return s.replace("-\n", "").replace("\n", " ")


class CitationExtracter :
	
	def __init__(self):
		
		self.txt_folder  = "/data/txt"
		self.matched_format_limit = 10
		
		self.build_regexes()
		

	def build_regexes(self) :
		'''
		Construct, compile and return all implemented regular expressions.
		'''

		# Numerical references enclosed by brackets.
		# [1], [17], [1,2,3] 
		NUMERIC = '\[((\d+|\d+-\d+)(,\s*)?)+\]'

		# Alphanumeric references ending in digits (year) enclosed by brackets. 
		ALPHANUMERIC = '\[([a-z]+\d{2}[a-z]?(,\s*)?)+\]'

		# Authors followed by the year (optionally separated by a comma), all enclosed by parenthesis.
		# (Zaki, 2001), (Sequeira and Zaki, 2004b), (Friedman and Meulman, 2004; Jing et al., 2007), 
		# Huang (1997), Jing et al. (2007), Ahmad and Dey (2007b).
		authors = '([\w\s.,&]+)'
		year = '(20|19)\d{2}([a-z](,\s*)?)*'
		pattern1 = '\((%s,?\s*%s(;\s)?)+\)' % (authors, year)

		first_author = '[a-z]+'
		two_authors = '([a-z]+\s+(and|&)\s+[a-z]+)'
		etal = '(\s+et\s+al.?)?'
		pattern2 = '(' + two_authors + '|' + '(' + first_author + etal + '))\s*\(' + year + '?\)' 
		AUTHORS_AND_YEAR = "(%s)|(%s)" % (pattern1, pattern2)

		self.regexes = [("numeric", re.compile(NUMERIC, re.IGNORECASE)),
										("alphanumeric", re.compile(ALPHANUMERIC, re.IGNORECASE)),
										("authors_year", re.compile(AUTHORS_AND_YEAR, re.IGNORECASE))]

		# Regular expressions for splitting references
		self.ref_regexes = [re.compile("(\[\d+\])", re.IGNORECASE),	
												re.compile("((\n|^)+\s*\d+\s*\.)", re.IGNORECASE)]


	def parse_cited_numeric(self, citation_str):

		cited = []

		citation_str = citation_str[1:-1]  # Remove surrounding brackets []
		cids = map(string.strip, citation_str.split(","))
		for cid in cids :
			
			# Check if the range kind of e.g. [1-4] or not
			if cid.find('-')<0 :
				cited.append(cid)
			else :
				start, end = cid.split('-')
				cited_range = range(int(start.strip()), int(end.strip())+1)
				cited.extend(map(str, cited_range))

		return cited


	def get_citations(self, text):
	
		format_count = defaultdict(int)
		cit_loc = []
		cit_str = []

		for (format, regex) in self.regexes:

			for match in regex.finditer(text) :

				start, end = match.span()
				cit_loc.append((start, end))
				cit_str.append(text[start:end])

				format_count[format] += 1
			
			# If we found a reasonable number of citations we declare that
			# the format is defined and skip searching for the other ones.
			if format_count[format] >= self.matched_format_limit :
				break


		if len(cit_loc)==0 :
			raise NoCitationsFound()

		format, _count = max(format_count.items(), key=(lambda (k,v): v))
		return format, cit_loc, cit_str


	def get_ref_section(self, text):

		beg_ref_section = re.compile("(^|\W)(references|bibliography)($|\W)", re.IGNORECASE)
		end_ref_sections = re.compile('appendix|acknowledgements', re.IGNORECASE)

		begs = list(re.finditer(beg_ref_section, text))
		
		if len(begs)==0 :
			raise ReferencesNotFound()
		
		# Get last occurrence of the regex as the beginning of the reference section
		begin = begs[-1].start()
		
		# Search for possible end points for the reference section
		ends = list(re.finditer(end_ref_sections, text[begin:]))

		# Get the first candidate for references ending. If 
		# none found, just use None for the end of the string.
		if ends :
			end = begin+ends[0].start()
			ref_str = text[begin:end]
			non_ref_str = text[:begin] + text[end:]
		else:
			ref_str = text[begin:]
			non_ref_str = text[:begin]
			
		return begin, ref_str, non_ref_str
	

	def split_ref_by_regex(self, refs_text, regexes):

		# Store the position (start:end) and string value of each reference entry
		refs_pos = []
		refs_str = []

		for regex in regexes :
			matches = [m.start() for m in regex.finditer(refs_text)]
			if matches :
				break

		# Raise exception if no regex matched something
		if not matches:
			raise ReferencesNotFound()

		# To handle last interval (None stands for the end of the document)
		matches.append(None)
		for i in xrange(len(matches)-1):

			s = matches[i]
			e = matches[i+1]

			refs_pos.append((s,e))
			refs_str.append(refs_text[s:e])

		# Append the last one
# 		s = matches[-1].start()
# 		refs_pos.append((s,None))
# 		refs_str.append(refs_text[s:])

		return refs_pos, refs_str


	def show_citations(self, txt_file, citations) :
		
		with open(txt_file, 'r') as f :
			text = f.read()
			
		for start, end, cited in citations :
			print text[start:end], str(cited)


	def exact_citations_match(self, citation_ids, refs):
		cit_to_ref = {}
		for cid in citation_ids :

			for i, ref in enumerate(refs):
				if ref.strip().startswith("[%s]"%cid) :
					cit_to_ref[cid] = i
					break
				
		return cit_to_ref
	
			
	def merge_cited_ids(self, citations_cited):
		''' Get unique ids. '''
		cited = set()
		cited.update(*citations_cited)
		return cited
	
		
	def sentence_limit(self, text, start, direction):
		'''
		Gets the limit for the sentence containing the given indexes.
		'''
		
		if direction==1 :
			end = len(text)
		elif direction==-1:
			end = 0
		else :
			raise TypeError("Unsupported direction value passed: %d." % direction)
	
		curr = start+direction
		next = curr+direction
		while (curr!=end) :
	
			if text[curr] == '.' :
				break
	
			curr += direction
		
		return curr+1


	def get_contexts(self, text, positions) :

		sentences = []
		for (start, end) in positions :
			ss = self.sentence_limit(text, start, -1)
			se = self.sentence_limit(text, end-1, 1)
#			sentences.append( clean(text[ss:start] + text[end:se]) )
			sentences.append( clean(text[ss:se]) )
				
		return sentences


	def get_citations_contexts(self, txt_file) :
		'''
		Receives a paper txt parses citation related content:
		  - Finds the citations in the text.
		  - Parses the references section (splitting each reference).
		  - Finds the context around each citation in the text (sentence containing the citation).
		  
		Returns:
		  cits_pos: Array with indexes of the citations in the text (start_idx, end_idx).  
		  cits_str: Array of same dimension with the strings of the citations (e.g.: '[13, 2]').
		  contexts: Array of same dimension with citation contexts.
		  cits_to_ref_strs: Dictionary with the reference identifiers (13, 2, etc.) and 
		                    their corresponding reference strings.
		'''

		try :

			with open(txt_file, 'r') as f :
				text = f.read()

			# Split the references section from the rest (no sense in detecting citations in the references section).
			_refs_sec_start, refs_str, non_refs_str = self.get_ref_section(text)
			
			# Detect citations in the text
			format, cits_pos, cits_str = self.get_citations(non_refs_str)

			if format == "numeric" :

				# Parse each citation string to get the cited identifiers
				cits_cited = map(self.parse_cited_numeric, cits_str)

				# Merge all the cited ids into a single set of unique ids
				cited_ids = self.merge_cited_ids(cits_cited)

				# Split each individual references in the references section
				_refs_loc, refs_str = self.split_ref_by_regex(refs_str, self.ref_regexes)

				# Match the citation identifier (used in the text) to the corresponding reference.
				cits_to_refs = self.exact_citations_match(cited_ids, refs_str)

				# Now match reference ids to their corresponding text
				cits_to_ref_strs = {id: refs_str[idx] for id, idx in cits_to_refs.items()}
				
				# Get contexts from positions
				contexts = self.get_contexts(text, cits_pos)

				# Save everything into the DB
			else :
				print "%s: Citation format %s not supported yet." % (txt_file, format)

		# Nothing to collect
		except ReferencesNotFound:
			print "References section not found."

		except NoCitationsFound:
			print "No citations found."


		return cits_pos, cits_str, contexts, cits_to_ref_strs


if __name__ == '__main__':

	txt_file = "1.txt"

	print "Finding citations and contexts for '%s'.\n" % txt_file

	extractor = CitationExtracter()
	cpos, cstr, ctxs, cits_to_refs = extractor.get_citations_contexts(txt_file)
	for i in xrange(len(cpos)) :
		s, e = cpos[i]
		print "%-16s\t%s" % (cstr[i], ctxs[i])


