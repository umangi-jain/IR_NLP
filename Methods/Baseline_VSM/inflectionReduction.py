from util import *

# Add your import statements here
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer 
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 



class InflectionReduction:

	def reduce(self, text):
		"""
		Stemming/Lemmatization

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed/lemmatized tokens representing a sentence
		"""
		#lemmatization is the preferred choice for IR
		reducedText = []
		
		lemmatizer = WordNetLemmatizer() 
		
		for i in text: # i is one sentence
			redsent = []
			for j in i: #j is a word
				redsent.append(lemmatizer.lemmatize(j))
			reducedText.append(redsent)
				
		# ps = PorterStemmer()
		# for i in text: # i is one sentence
		# 	for j in i: #j is a word
		# 		reducedText.append(ps.stem(j))
				#reducedText.append(j)


		#Fill in code here
		
		return reducedText


