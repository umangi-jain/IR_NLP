from util import *
import numpy as np
import csv
import pickle
from collections import Counter
from nltk.corpus import brown
from mittens import GloVe, Mittens
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import CountVectorizer

# Add your import statements here


class InformationRetrieval():

	def __init__(self):
		self.index = None	# Stores the inverted index representation for the documents, which is built in the buildIndex() funciton
		self.N = 0 # Stores the total number of documents
		self.terms = set([]) # stores the terms present in documents and in queries, which would form the terms that form the dimentions in the vector space
		self.docIDs = [] # stores the document ids of all the documents. It is initialised from the docIDs list passed in the buildIndex function and used in other class functions.
		self.index_terms = {}
		self.alldocs = []
	def buildIndex(self, docs,titles, docIDs):
		
		self.N = len(docs)
		self.docIDs = docIDs
		for doc in docs:
			for sentence in doc:
				for word in sentence:
					self.terms.add(word.lower())
		for title in titles:
			for sentence in title:
				for word in sentence:
					self.terms.add(word.lower())

		print("start here")
		for doc in docs:
			for sentence in doc:
				for word in sentence:
					self.alldocs.append(word)
		print("end here")
		i =0
		for term in self.terms:
			self.index_terms[term] = i
		
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
		
		tindex = {}

		for term in self.terms:
			tindex[term] = []

			for i in range(len(titles)):
				tf = 0
				for sentence in titles[i]:
					for word in sentence:
						if word.lower() == term:
							tf += 1
				# the entry for a document is included in index[term] only if the term is present atleast once in the in that doc (tf > 0)
				if tf>0:
					tindex[term].append((docIDs[i],tf))

			# arranging the list of tuples in decreasing order of tf of the term in the documents
			tindex[term] = sorted(tindex[term], key = lambda a: a[1], reverse=True)

		self.tindex = tindex


		

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

		# Calculating the TF-IDF vector of each document
		for docId in self.docIDs:
			tf_idf_vectors_of_docs[docId] = np.zeros(self.dim,dtype='float')

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


							tf_idf_vectors_of_docs[docId] += self.my_dict[term]*tf*idf
							

			
				i += 1
			

		return tf_idf_vectors_of_docs
	def getTF_IDFVectorsOfTitles(self,idf_of_terms):
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

		tf_idf_vectors_of_titles = {} # Stores the entity to be returned, as described in the description of return type above
		numOfTerms = len(self.terms) # Stores the number of terms present self.terms

		# Calculating the TF-IDF vector of each document
		for docId in self.docIDs:
			tf_idf_vectors_of_titles[docId] = np.zeros(self.dim,dtype='float')

			# Calculating the TF-IDF weight of each term
			i = 0
			for term in self.terms:
				tf = idf = 0
				idf = idf_of_terms[term]

				# finding the tf of the present term in the present document (tf = 0 , if a term is not present in the document) 
				if term in self.tindex.keys():
					for item in self.tindex[term]:
						if docId == item[0]:
							tf = item[1]
							tf_idf_vectors_of_titles[docId] += self.my_dict[term]*tf*idf

				
				i += 1

		return tf_idf_vectors_of_titles


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
		tf_idf_vector = np.zeros(self.dim,dtype='float') # Stores the entity to be returned, as described in the description of return type above

		# Calculating the TF-IDF vector of query
		i = 0
		for term in self.terms:
			tf = 0

			# finding the tf of the present term in the query (tf = 0 , if a term is not present in the document) 
			for sentence in query:
				for word in sentence:
					if word.lower() == term:
						tf += 1

			tf_idf_vector += tf*idf_of_terms[term]*self.my_dict[term]
			i += 1
		#print(tf_idf_vector)
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
		i = 0
		for term in self.terms:
			i = i+1
			if term in self.index.keys(): # term present in atleast one of the documents
				n = len(self.index[term]) # n is the number of documents in which that term is present.
				if (n == 0):
					idf =0
				else:
					idf = np.log10(self.N/n)

			else:
				idf = 0

			idf_of_terms[term] = idf

		unknown = 0
		self.my_dict = {}
		print(len(self.terms))
		
		self.dim = 300
		kkl = 0
		with open("glove.42B.300d.txt", 'r') as f:
			for line in f:
				values = line.split()
				word = values[0]
				kkl = kkl+1

				if(kkl%1e9 == 0):
					print(kkl)
				if word.lower() == "unk":
					aimp = np.asarray(values[1:], "float32")
				if word.lower() in self.terms:

				    vector = np.asarray(values[1:], "float32")
				    self.my_dict[word.lower()] = vector
			# else:A
			# 	self.my_dict[word.lower()] = -100
			# 	unknown = unknown +1



		corp_vocab = []
		print(unknown,kkl)
		for term in self.terms:
			if term.lower() in self.my_dict.keys():
				pass
			else: 
				unknown += 1
				corp_vocab.append(term.lower())
				# self.my_dict[term.lower()] = aimp
		 	# if(self.my_dict[term.lower()] == -100):
	 			# ;
		 	

		print("unknown words ", len(corp_vocab))
		#print(self.embeddings_dict["unk"])
		cv = CountVectorizer(ngram_range=(1,1), vocabulary=corp_vocab)
		X = cv.fit_transform(self.alldocs)
		Xc = (X.T * X)
		Xc.setdiag(0)
		coocc_ar = Xc.toarray()
		np.shape(coocc_ar)
		print("mitten begin")
		mittens_model = Mittens(n=300, max_iter=500)
		new_embeddings = mittens_model.fit(
		    coocc_ar,
		    vocab=corp_vocab,
		    initial_embedding_dict= self.my_dict)
		print("mittens end")
		newglove = dict(zip(corp_vocab, new_embeddings))
		f = open("repo_glove.pkl","wb")
		
		self.my_dict = {**self.my_dict, **newglove}
		pickle.dump(self.my_dict, f)
		f.close()

		simeval_pairs = []
		simeval_scores = []
		this_scores = []
		with open('/home/umangi/Docs/Semester8/NLP/project/wordsim evaluation/simlex_common.csv', 'r') as sim1:
		    reader = csv.reader(sim1)
		    for row in reader:
		    	simeval_pairs.append([row[0], row[1]])
		    	simeval_scores.append(row[2])
		        
		for rr in range(len(simeval_scores)):
			x = simeval_pairs[rr][0].lower()
			y = simeval_pairs[rr][1].lower()
			tr = np.dot(self.my_dict[x], self.my_dict[y])
			this_scores.append(tr)
		print("simlex ", len(simeval_pairs))  
		with open('jaccard_simeval.csv', mode='w') as employee_file:
		    employee_writer = csv.writer(employee_file, delimiter=',')
		    for pl in range(len(simeval_pairs)):
		    	employee_writer.writerow([simeval_pairs[pl][0], simeval_pairs[pl][1], str(simeval_scores[pl]), str(this_scores[pl])])
		##############
		simeval_pairs = []
		simeval_scores = []
		this_scores = []
		with open('/home/umangi/Docs/Semester8/NLP/project/wordsim evaluation/wordsim_common.csv', 'r') as sim1:
		    reader = csv.reader(sim1)
		    for row in reader:
		    	simeval_pairs.append([row[0], row[1]])
		    	simeval_scores.append(row[2])
		print("wordsim ", len(simeval_pairs))        
		for rr in range(len(simeval_scores)):
			x = simeval_pairs[rr][0].lower()
			y = simeval_pairs[rr][1].lower()
			tr = np.dot(self.my_dict[x], self.my_dict[y])
			this_scores.append(tr)

		with open('jaccard_wordsim.csv', mode='w') as employee_file:
		    employee_writer = csv.writer(employee_file, delimiter=',')
		    for pl in range(len(simeval_pairs)):
		    	employee_writer.writerow([simeval_pairs[pl][0], simeval_pairs[pl][1], str(simeval_scores[pl]), str(this_scores[pl])])










		# Getting the TF-IDF vector representations i=of the documents
		tf_idf_vectors_of_docs = self.getTF_IDFVectorsOfDocs(idf_of_terms )
		tf_idf_vectors_of_titles = self.getTF_IDFVectorsOfTitles(idf_of_terms)
		# Calculating the norm of TF-IDF vectors representing the documents




		for i in range(len(tf_idf_vectors_of_docs)):
			tf_idf_vectors_of_docs[i+1] = tf_idf_vectors_of_docs[i+1]+0*tf_idf_vectors_of_titles[i+1]
		doc_norms = {} # Key : doc id and value : norm of the TF-IDF vector of the document
		for docId in self.docIDs:
			doc_norms[docId] = np.linalg.norm(tf_idf_vectors_of_docs[docId])

		title_norms = {} # Key : doc id and value : norm of the TF-IDF vector of the document
		for docId in self.docIDs:
			title_norms[docId] = np.linalg.norm(tf_idf_vectors_of_titles[docId])


		doc_IDs_ordered = []	# Stores the entity to be returned, as described in the description of return type above
		titles_IDs_ordered = []
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
				if doc_norms[docId]==0 or query_norm==0 : 
				# If either the norm of the document or the norm of the query os 0, them the similarity is defined as 0 (as explained the the 
				# assumptions section in the report)
					sim = 0
				else:
					# cosine similarity measure = (tf-idf vector of doc).(tf-idf vector of query) / (norm(doc).norm(query)) 
					sim = np.dot(tf_idf_vector_of_query,tf_idf_vectors_of_docs[docId])/(query_norm*doc_norms[docId])
					# sim2 = np.dot(tf_idf_vector_of_query,tf_idf_vectors_of_titles[docId])/(query_norm*title_norms[docId])
					# sim = 2*sim1*sim2/(sim1+sim2)
				cosine_similarity_scores.append((docId,sim))

			# Sorting the cosine_similarity_scores list in decreasing order of similarity value
			cosine_similarity_scores = sorted(cosine_similarity_scores, key = lambda a: a[1], reverse=True)
	
			# Collecting the document ids arranged in decresing order of similarity measure
			doc_IDs_ordered_i = [] # stores the list of document ids ranked in decrwasing order of similarity measures for query i
			for item in cosine_similarity_scores:
				doc_IDs_ordered_i.append(item[0])

			doc_IDs_ordered.append(doc_IDs_ordered_i)
		
		return doc_IDs_ordered
