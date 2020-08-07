from util import *
import numpy as np
import pickle
import json
import statistics
# Add your import statements here


class InformationRetrieval():

	def __init__(self):
		self.index = None	# Stores the inverted index representation for the documents, which is built in the buildIndex() funciton
		self.N = 0 # Stores the total number of documents
		self.terms = set([]) # stores the terms present in documents and in queries, which would form the terms that form the dimentions in the vector space
		self.docIDs = [] # stores the document ids of all the documents. It is initialised from the docIDs list passed in the buildIndex function and used in other class functions.
		self.docs  = []
	def buildIndex(self, docs,titles, docIDs):
		self.N = len(docs)
		self.docIDs = docIDs

		for i in range(len(titles)):
			for j in range(len(titles[i])):
				docs[i].append(titles[i][j])
				
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


	
	def rank(self, queries, qrels):
		dlens = []
		for doc in self.docs:
			clen = 0
			for sent in doc:
				for word in sent:
					clen += 1
			dlens.append(clen)
		avglen = statistics.mean(dlens)
		print("avergae doc length ", avglen)

		for query in queries:
			for sentence in query:
				for word in sentence:
					self.terms.add(word.lower())

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


		xquer_order, xtrue_ids = extract_partial(qrels,0)#query id as 2,4,6
		print("len ", len(xtrue_ids), xtrue_ids[0])
		# allreldocs = set([])
		# for i in xtrue_ids:
		# 	for j in i:
		# 		allreldocs.add(j[0])
		# print("all relevant docs ", len(allreldocs))
		# allrelterms = set([])
		# for iid in allreldocs:
		# 	dco = self.docs[iid-1]
		# 	for sen in dco:
		# 		for wo in sen:
		# 			allrelterms.add(wo.lower())
		# print("all relevant words ",len(allrelterms))
		# allrelterms2 = set([])
		# for i in xquer_order:
		# 	qq = queries[i-1]
		# 	for sen in qq:
		# 		for wo in sen:
		# 			allrelterms2.add(wo.lower())
		# print("set2 ", len(allrelterms2))
		# print("inter ",len(allrelterms2.intersection(allrelterms)) )


		yquer_order, ytrue_ids = extract_partial(qrels,1)
		N = len(self.docs)
		pweights = {}
		weights = {}
		count = 0
		print(len(self.terms), len(self.docs))
		how_relevant = 0
		for kword in self.terms:
			isrel = 0
			# print(kword)
			if(count%1000 == 0):
				print(count)
			count +=1
			pweights[kword] = []
			n= 0 # number of doc containing the word
			avgr = 0 # number of relevant docs containing the word
			avgR = 0 #number of documents


			for doc in self.docs:
				nn = 0
				for sentence in doc:
					for word in sentence:
						if (word == kword):
							nn = 1
							break
				if (nn == 1):
					n +=1
			pweights[kword].append(n)
			all_p = []
			all_np = []
			for q in range(1,len(queries)+1):
				R = 0
				r = 0
				# print(q)
				if(q%2 == 0):
					query = queries[q-1]
					# for sent in query:
					# 	for w in sent:
					# 		if(w == kword):
								# print("q", q)
								
					for k in range(len(xquer_order)):
						# print("comp ", q, xquer_order[k])
						if(xquer_order[k] == q):
							# print("query id")
							
							rel_docs = xtrue_ids[k]
							R = len(rel_docs)
							for p in rel_docs:
								p = p[0]
								rr = 0
								for sentence in self.docs[p-1]:
									for word in sentence:
										if(word == kword):
											rr = 1
											isrel = 1
											break
								if(rr == 1):
									r +=1
							break

						
					all_p.append((r+0.1)/(R+0.1))
					all_np.append((n-r+0.1)/(N-R+0.1))
			pweights[kword].append(r)
			pweights[kword].append(R)
			if (isrel == 1):
				how_relevant += 1
			p1 = statistics.mean(all_p)
			p2 = statistics.mean(all_np)
			if(p1 == 0 or p2 ==0):
				print(p1,p2)
				print(all_p, all_np)
			# weights[kword] = np.log(((r+0.5)*(N-n-R+r+0.5))/((R-r+0.5)*(n-r+0.5)))
			weights[kword] = np.log((p1*(1-p2)+0.01)/(p2*(1-p1)+0.01))
			
		# stop
		a_file = open("data.json", "w")
		print("number of relevnt ", how_relevant)

		json.dump(pweights, a_file)

		a_file.close()





		k1 = 1.2
		b = 0.75
		doc_IDs_ordered = []	# Stores the entity to be returned, as described in the description of return type above

		# # Ranking the documents for each query in decreasing order of cosine similarity measures.
		
		for i in range(len(queries)):
			if(i%2 == 0):
				if(i%50 == 0):
					print(str(i)+ "queries completed")
				query = queries[i]




				similarity_scores = [] # each entry is a tuple -> (docId, similarity_measure)
				for ll in range(len(self.docs)):
					docId = self.docIDs[ll]
					doc = self.docs[ll]
					s = 0
					cons = 1-b+b*(dlens[ll]/avglen)
					k2 = k1*cons
					for sentence in query:
						for word in sentence:
							found = 0
							for sen2 in doc:
								for wor2 in sen2:
									if(wor2 == word):
										found += 1
										# s += weights[word] #+ 2*idf_of_terms[word]
							s += found*(k1+1)*(weights[word]+2*idf_of_terms[word])/(k2+found) 
										# s  += idf_of_terms[word]
					similarity_scores.append((docId,s))



				# Sorting the cosine_similarity_scores list in decreasing order of similarity value
				cosine_similarity_scores = sorted(similarity_scores, key = lambda a: a[1], reverse=True)
		
				# Collecting the document ids arranged in decresing order of similarity measure
				doc_IDs_ordered_i = [] # stores the list of document ids ranked in decrwasing order of similarity measures for query i
				for item in cosine_similarity_scores:
					doc_IDs_ordered_i.append(item[0])

				doc_IDs_ordered.append(doc_IDs_ordered_i)
			
		return doc_IDs_ordered