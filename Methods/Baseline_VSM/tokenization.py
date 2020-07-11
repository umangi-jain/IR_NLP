from util import *

# Add your import statements here
from nltk.tokenize import TreebankWordTokenizer

class Tokenization():

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""
		#print(text)
		tokenizedText = []
		for k in text: #look at each entity in one sentence
			
			a = ""#stores the current word 
			run = []; #appends all words in a particular sentence
			for i in range(len(k)):
				
				if(k[i] == ' ' or k[i] == '	'): #tokenization at space or tab
					
					if(a!=""):
						if(a[-1] == ',' or a[-1] == '-' or a[-1] == "\'" or a[-1] == ";" or a[-1] == ":" or a[-1] =="!" or a[-1] == "?" or a[-1] =="\"") : #but remove mentioned punctuations from the end of the word, if present
							a = a[:-1]
						if(len(a)>0 and a[0] == "\""):#remove starting quotes
							a = a[1:]
						if(len(a)>0):
							run.append(a)
							
							a = ""


				elif(i == len(k)-1): #remove the last punctuation mark, if present
					
					a = a+k[i];
					
					if(a[-1] == '.' or a[-1] == '\"' or a[-1] =="!" or a[-1] == "?" or a[-1] =="\'" ):
						a = a[:-1]
					if(len(a)>0 and a[0] == "\""):
						a = a[1:]
					if(len(a)>0):
						run.append(a)
						
						a = ""


				else:
					
					if((k[i] == ',' or k[i] == ':' or k[i] == ';') and k[i+1]!= ' ' ): # for other punctuation marks followed by a space
						#print(k[i-1])
						if(len(a)>0):
							if(a[-1] == '\"' or a[-1] =="!" or a[-1] == "?" ):
								a = a[:-1]
							if(len(a)>0 and a[0] == "\""):
								a = a[1:]
							if(len(a)>0):
								run.append(a)
								
								a = ""


					else:

						a = a+k[i];

			tokenizedText.append(run)		

		
			




		#Fill in code here

		return tokenizedText



	def pennTreeBank(self, text):
		
		tokenizedText = []

		
			
		for j in text:
					
			a = TreebankWordTokenizer().tokenize(j)
			tokenizedText.append(a)
		
			
			
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		

		#Fill in code here

		return tokenizedText


