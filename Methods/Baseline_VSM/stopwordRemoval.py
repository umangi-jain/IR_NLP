from util import *

# Add your import statements here
import nltk
from nltk.corpus import stopwords




class StopwordRemoval():

	def fromList(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""

		#Getitng the list of stopwords from nltk
		stopWords = set(stopwords.words('english'))

		stopwordRemovedText = []

		for sentence in text:
			stopwordRemovedSentence = []

			# appending a word to stopwordRemovedSentence only if it is not a stopword
			for word in sentence:
				if word not in stopWords:
					stopwordRemovedSentence.append(word)

			stopwordRemovedText.append(stopwordRemovedSentence)

		return stopwordRemovedText


	
