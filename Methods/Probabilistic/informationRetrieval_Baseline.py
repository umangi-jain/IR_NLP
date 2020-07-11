from util import *
import numpy as np
# Add your import statements here


class InformationRetrieval():

	def __init__(self):
		self.index = None	# Stores the inverted index representation for the documents, which is built in the buildIndex() funciton
		self.N = 0 # Stores the total number of documents
		self.terms = set([]) # stores the terms present in documents and in queries, which would form the terms that form the dimentions in the vector space
		self.docIDs = [] # stores the document ids of all the documents. It is initialised from the docIDs list passed in the buildIndex function and used in other class functions.
		self.docs  = []
	def buildIndex(self, docs, docIDs):
		self.N = len(docs)
		self.docIDs = docIDs
		self.docs = docs
		# Collecting all the terms in the documents into the set self.terms
		for doc in docs:
			for sentence in doc:
				for word in sentence:
					self.terms.add(word.lower())
		index = {}

		for term in self.terms:
			index[term] = []

			for i in range(len(docs)):
				tf = 0
				for sentence in docs[i]:
					for word in sentence:
						if word.lower() == term:
							tf += 1
				# the entry for a document is included in index[term] only if the term is present atleast once in the in that doc (tf > 0)
				if tf>0:
					index[term].append((docIDs[i],tf))

			# arranging the list of tuples in decreasing order of tf of the term in the documents
			index[term] = sorted(index[term], key = lambda a: a[1], reverse=True)

		self.index = index


	def getTF_IDFVectorsOfDocs(self,idf_of_terms):
	

		tf_idf_vectors_of_docs = {} # Stores the entity to be returned, as described in the description of return type above
		numOfTerms = len(self.terms) # Stores the number of terms present self.terms

		# Calculating the TF-IDF vector of each document
		for docId in self.docIDs:
			tf_idf_vectors_of_docs[docId] = np.zeros(numOfTerms,dtype='float')

			# Calculating the TF-IDF weight of each term
			i = 0
			for term in self.terms:
				tf = idf = 0
				idf = idf_of_terms[term]

				# finding the tf of the present term in the present document (tf = 0 , if a term is not present in the document) 
				if term in self.index.keys():
					for item in self.index[term]:
						if docId == item[0]:
							tf = item[1]

				tf_idf_vectors_of_docs[docId][i] = idf
				i += 1

		return tf_idf_vectors_of_docs


	def getTF_IDFVectorOfQuery(self,query, idf_of_terms):
		
		numOfTerms = len(self.terms) # Stores the number of terms present self.terms
		tf_idf_vector = np.zeros(numOfTerms,dtype='float') # Stores the entity to be returned, as described in the description of return type above

		# Calculating the TF-IDF vector of query
		i = 0
		for term in self.terms:
			tf = 0

			# finding the tf of the present term in the query (tf = 0 , if a term is not present in the document) 
			for sentence in query:
				for word in sentence:
					if word.lower() == term:
						tf += 1

			tf_idf_vector[i] = idf_of_terms[term]
			i += 1

		return tf_idf_vector


	def rank(self, queries):
		
		for query in queries:
			for sentence in query:
				for word in sentence:
					self.terms.add(word.lower())

		# Computing the idf of the terms
		idf_of_terms = {}
		# If the term is present in atleast one of the documents the IDF is calculated as np.log10(N/n), where N is totla number of documents
		# and n is the number of documents in which that term is present.
		# if not, the idf of the term is defined as 0
		for term in self.terms:
			if term in self.index.keys(): # term present in atleast one of the documents
				n = len(self.index[term]) # n is the number of documents in which that term is present.
				idf = np.log10(self.N/n)
			else:
				idf = 0

			idf_of_terms[term] = idf

		# # Getting the TF-IDF vector representations i=of the documents
		# tf_idf_vectors_of_docs = self.getTF_IDFVectorsOfDocs(idf_of_terms)

		# # Calculating the norm of TF-IDF vectors representing the documents
		# doc_norms = {} # Key : doc id and value : norm of the TF-IDF vector of the document
		# for docId in self.docIDs:
		# 	doc_norms[docId] = np.linalg.norm(tf_idf_vectors_of_docs[docId])


		doc_IDs_ordered = []	# Stores the entity to be returned, as described in the description of return type above

		# # Ranking the documents for each query in decreasing order of cosine similarity measures.
		
		for i in range(len(queries)):
			if(i%50 == 0):
				print(str(i)+ "queries completed")
			query = queries[i]




			similarity_scores = [] # each entry is a tuple -> (docId, similarity_measure)
			for ll in range(len(self.docs)):
				docId = self.docIDs[ll]
				doc = self.docs[ll]
				s = 0
				for sentence in query:
					for word in sentence:
						for sen2 in doc:
							for wor2 in sen2:
								if(wor2 == word):
									s += idf_of_terms[word]
				similarity_scores.append((docId,s))



			# Sorting the cosine_similarity_scores list in decreasing order of similarity value
			cosine_similarity_scores = sorted(similarity_scores, key = lambda a: a[1], reverse=True)
	
			# Collecting the document ids arranged in decresing order of similarity measure
			doc_IDs_ordered_i = [] # stores the list of document ids ranked in decrwasing order of similarity measures for query i
			for item in cosine_similarity_scores:
				doc_IDs_ordered_i.append(item[0])

			doc_IDs_ordered.append(doc_IDs_ordered_i)
		
		return doc_IDs_ordered
