from util import *
import nltk
# Add your import statements here




class SentenceSegmentation():

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText = []
		i = 0
		l = len(text)

		while i<l:

			if text[i]=='.' or text[i]=='?' or text[i]=='!':

				# case 2 : If punctuation is not floowed by a space and " or uppercase letter, then it is not end of sentence at index i
				if i<l-1:
					if (not text[i+1]==' ') and (not text[i+1]=='\"') and (not text[i+1].isupper()): #not end of sentence
						i = i+1;
						continue

					#case 3: if '.' is fllowed by ", then the end of sentence is updated to after " and not after '.'
					if text[i+1]=='\"':
						i = i+2

				if i!=l-1:
					sentence ,remainingText = text[:i+1],text[i+1:]
					segmentedText.append(sentence)
					text = remainingText
				else:
					sentence = text[:i+1]
					segmentedText.append(sentence)
					text=""
		
				i = 0
				l = len(text)

			else:	#punctuation mark is not found
				i = i+1

		return segmentedText





	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each strin is a single sentence
		"""

		# Using punkt tokenizer from nltk package
		senetnce_tokenizer = nltk.data.load('tokenizers/punkt/PY3/english.pickle')
		segmentedText = senetnce_tokenizer.tokenize(text)

		return segmentedText
