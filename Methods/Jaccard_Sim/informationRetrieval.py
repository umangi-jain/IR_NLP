from util import *
import numpy as np
# Add your import statements here


class InformationRetrieval():

	def __init__(self):
		self.index = None	# Stores the inverted index representation for the documents, which is built in the buildIndex() funciton
		self.N = 0 # Stores the total number of documents
		self.terms = set([]) # stores the terms present in documents and in queries, which would form the terms that form the dimentions in the vector space
		self.docIDs = [] # stores the document ids of all the documents. It is initialised from the docIDs list passed in the buildIndex function and used in other class functions.

	def buildIndex(self, docs,titles, docIDs):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
		"""

		# Building the inverted index representation of the documents, and this build index is used to initialise self.index class variable 

		# Initialising the total number of documents present and the document ids of the documents in the class variables.
		self.N = len(docs)
		self.docIDs = docIDs
		self.vocab = []
		# Collecting all the terms in the documents into the set self.terms
		# for i in range(len(titles)):
		# 	for j in range(len(titles[i])):
		# 		docs[i].append(titles[i][j])
		# print("changed")
		for doc in docs:
			for sentence in doc:
				for word in sentence:
					self.terms.add(word.lower())




		# index is defined as a dictionary where key is term and value is a list of tuples, where we have one tuple corresponding to each 
		# document in which that term occurs atleast once. Each tuple is of the form (docID, tf) -> where docID is the document of that 
		# document and tf is th term frequency of the term in the document. The list if tuples for each term are arranges in the decreasing
		# order of tf of the term in the documents.
		index = {}
		print("12")
		madness = 0
		for term in self.terms:
			if(madness%200 == 0):
				print(madness)
			madness += 1
			index[term] = []
			self.vocab.append(term)

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
		alp =0.5
		self.index = index
		print("Coocurence begins")
		print("len of terms = ", len(self.terms), len(self.index))
		print("vocab size = ", len(self.vocab))
		self.cooccur = np.eye(len(self.vocab))
		for i in range(len(self.vocab)):
			if(i%500 == 0):
				print(i)
			for j in range(i+1,len(self.vocab)):
				num = 0
				den = 0
				for item1 in self.index[self.vocab[i]]:
					for item2 in self.index[self.vocab[j]]:
						if item1[0] == item2[0]:
							num = num+1
				den = len(self.index[self.vocab[i]])+len(self.index[self.vocab[j]])-num
				if num>0 and den>0:
					self.cooccur[i][j] = num/den
					self.cooccur[j][i] = num/den
			self.cooccur[i] = np.exp(self.cooccur[i])/sum(np.exp(self.cooccur[i])) 
		print("Coocurence ends")
		print(np.shape(self.cooccur))
		f = 0
		for i in range(len(self.vocab)):
			for j in range(len(self.vocab)):
				if(self.cooccur[i][j] >0.2 and self.cooccur[i][j]<1):
					print("i= ", i,"j= ", j, "val = ",self.cooccur[i][j], "word1= " ,self.vocab[i], "word2= ", self.vocab[j])
					f = f+1
					break;
			if(f>0):
				break
		print("testing over coov")
	def getTF_IDFVectorsOfDocs(self,idf_of_terms):
		"""
		An additional fucntion added to calculate and return the tf-idf representation of the documents.

		Parameters
		----------
		arg1 : dictionary
			Keys : terms in the documents
			Values : IDF of that term i the documents

		Returns : dictionary
		-------
			Keys : Document ids of the documents
			Value : list corresponding the TF-IDF vector of that document -> where each element in the list is 
			TF-IDF weight of the term, considering the terms in the order in which they are stores in self.terms(set). 
		"""

		tf_idf_vectors_of_docs = {} # Stores the entity to be returned, as described in the description of return type above
		numOfTerms = len(self.terms) # Stores the number of terms present self.terms
		print("verify")
		print(len(self.terms), len(self.vocab))
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
							for j in range(len(self.vocab)):
								# if(i>8838 or j >8838):
								# 		print("Error bounds", i,j)
								if(i<len(self.vocab)):
									if(self.cooccur[i][j]>0 and self.cooccur[i][j]<1):
										tf_idf_vectors_of_docs[docId][j] += self.cooccur[i][j]*tf*idf
									# for k in range(len(self.vocab)):
									# 	if(self.cooccur[j][k]>0 and self.cooccur[j][k]<1 and k!=i):
									# 		tf_idf_vectors_of_docs[docId][k] += self.cooccur[i][j]*self.cooccur[j][k]*tf*idf



				tf_idf_vectors_of_docs[docId][i] += tf*idf
				i += 1

		return tf_idf_vectors_of_docs


	def getTF_IDFVectorOfQuery(self,query, idf_of_terms):
		"""
		An additional fucntion added to calculate and return the tf-idf representation of the query.

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sentence of the query
		arg2 : dictionary
			Keys : terms in the documents
			Values : IDF of that term i the documents

		Returns : list
		-------
			list corresponding the TF-IDF vector of that query -> where each element in the list is 
			TF-iDF weight of the term, considering the terms in the order in which they are stores in self.terms(set). 
		"""
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

			tf_idf_vector[i] += tf*idf_of_terms[term]
			if(tf>0):
				for j in range(len(self.vocab)):
					if(i<len(self.vocab)):
						if(self.cooccur[i][j]>0 and self.cooccur[i][j]<1):
							tf_idf_vector[j] += self.cooccur[i][j]*tf*idf_of_terms[term]
						# for k in range(len(self.vocab)):
						# 	if(self.cooccur[j][k]>0 and self.cooccur[j][k]<1 and k!=i):
						# 		tf_idf_vector[k] += self.cooccur[i][j]*self.cooccur[j][k]*tf*idf


			i += 1

		return tf_idf_vector


	def rank(self, queries):
		"""
		Rank the documents according to relevance for each query

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query


		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		# Collecting all the terms in the queries into the set self.terms
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

		# Getting the TF-IDF vector representations i=of the documents
		tf_idf_vectors_of_docs = self.getTF_IDFVectorsOfDocs(idf_of_terms)

		# Calculating the norm of TF-IDF vectors representing the documents
		doc_norms = {} # Key : doc id and value : norm of the TF-IDF vector of the document
		for docId in self.docIDs:
			doc_norms[docId] = np.linalg.norm(tf_idf_vectors_of_docs[docId])


		doc_IDs_ordered = []	# Stores the entity to be returned, as described in the description of return type above

		# Ranking the documents for each query in decreasing order of cosine similarity measures.
		for i in range(len(queries)):
			query = queries[i]
			# Getting the TF-IDF vector of the query
			tf_idf_vector_of_query = self.getTF_IDFVectorOfQuery(query, idf_of_terms)
			# Calculating the norm of TF-IDF vector representing the query
			query_norm = np.linalg.norm(tf_idf_vector_of_query)

			# calculating the cosine similarity scores between the query and all the documents
			cosine_similarity_scores = [] # each entry is a tuple -> (docId, similarity_measure)
			for docId in self.docIDs:
				if doc_norms[docId]==0 or query_norm==0: 
				# If either the norm of the document or the norm of the query os 0, them the similarity is defined as 0 (as explained the the 
				# assumptions section in the report)
					sim = 0
				else:
					# cosine similarity measure = (tf-idf vector of doc).(tf-idf vector of query) / (norm(doc).norm(query)) 
					sim = np.dot(tf_idf_vector_of_query,tf_idf_vectors_of_docs[docId])/(query_norm*doc_norms[docId])

				cosine_similarity_scores.append((docId,sim))

			# Sorting the cosine_similarity_scores list in decreasing order of similarity value
			cosine_similarity_scores = sorted(cosine_similarity_scores, key = lambda a: a[1], reverse=True)
	
			# Collecting the document ids arranged in decresing order of similarity measure
			doc_IDs_ordered_i = [] # stores the list of document ids ranked in decrwasing order of similarity measures for query i
			for item in cosine_similarity_scores:
				doc_IDs_ordered_i.append(item[0])

			doc_IDs_ordered.append(doc_IDs_ordered_i)
		
		return doc_IDs_ordered
