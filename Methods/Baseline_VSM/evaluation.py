from util import *
import numpy as np
import math
# Add your import statements here




class Evaluation():

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		precision = -1

		#Fill in code here
		true_doc_IDs = np.asarray(true_doc_IDs)

		is_rel = 0 #number of relevant documents
		for i in range(k):
			for j in range(len(true_doc_IDs)):
				if query_doc_IDs_ordered[i] == true_doc_IDs[j][0]: #check if doc is true relevant doc
					is_rel = is_rel +1
					break;



	
			

		precision = is_rel/k

		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""

		meanPrecision = -1

		#Fill in code here
		meanPrecision = 0
		quer_order, true_ids = extract_true(qrels) #extract true_doc IDs
		for j in range(len(query_ids)):
			for l in range(len(quer_order)):
				if(quer_order[l] == query_ids[j]): #for a particular query
					meanPrecision = meanPrecision + self.queryPrecision(doc_IDs_ordered[j],query_ids[j],true_ids[l], k)
					break;
				


		#Fill in code here
		
		meanPrecision = meanPrecision/len(query_ids) #average over all queries

		return meanPrecision

	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""

		recall = -1

		#Fill in code here
		is_rel  = 0 #number of relevant docs
		total_ret = len(true_doc_IDs) #total number of relevant documents
		for i in range(k):
			for j in range(total_ret):
				if(query_doc_IDs_ordered[i] == true_doc_IDs[j][0]): #if relevnt doc found in ground truth
					is_rel = is_rel +1
					break
		

		recall = is_rel/total_ret

		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""

		meanRecall = -1

		#Fill in code here
		meanRecall = 0
		quer_order, true_ids = extract_true(qrels)
		for j in range(len(query_ids)):
			for l in range(len(quer_order)):
				if(quer_order[l] == query_ids[j]):
					meanRecall = meanRecall + self.queryRecall(doc_IDs_ordered[j],query_ids[j],true_ids[l], k)
					break;
				


		
		meanRecall = meanRecall/len(query_ids)

		return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""

		fscore = -1

		#Fill in code here
		p = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		r = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		if p == 0 or r == 0:
			fscore = 0
		else:
			fscore = 2*p*r/(p+r)

		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value
		
		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""

		meanFscore = -1

		#Fill in code here
		meanFscore = 0
		quer_order, true_ids = extract_true(qrels)
		for j in range(len(query_ids)):
			for l in range(len(quer_order)):
				if(quer_order[l] == query_ids[j]):
					meanFscore = meanFscore + self.queryFscore(doc_IDs_ordered[j],query_ids[j],true_ids[l], k)
					break;
				
		meanFscore = meanFscore/len(query_ids)

		return meanFscore
	

	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""

		nDCG = -1

		#Fill in code here
		true_doc_IDs = np.asarray(true_doc_IDs)
		p = np.argsort(true_doc_IDs[:,1])
		true_doc_IDs = true_doc_IDs[p] #sort ground truth wrt relevance
		
		
		idcg = 0 #ideal dcg (from ground truth)
		dcg = 0 #dcg obtained from vector space model

		for i in range(k):
			for j in range(len(true_doc_IDs)):
				if(query_doc_IDs_ordered[i] == true_doc_IDs[j][0]):
					dcg = dcg+(5-true_doc_IDs[j][1])/math.log2(i+2)
					break;
		t = min(k, len(true_doc_IDs))
		for j in range(t):
			
			idcg = idcg+(5-true_doc_IDs[j][1])/math.log2(j+2)
				
			# 1 means not very relevant and 4 meaning  highly relevant
		
		
		

		nDCG = dcg/idcg

		return nDCG


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""

		meanNDCG = -1

		#Fill in code here
		meanNDCG = 0
		quer_order, true_ids = extract_true(qrels)
		for j in range(len(query_ids)):
			for l in range(len(quer_order)):
				if(quer_order[l] == query_ids[j]):
					meanNDCG = meanNDCG + self.queryNDCG(doc_IDs_ordered[j],query_ids[j],true_ids[l], k)
					break;
				
		meanNDCG = meanNDCG/len(query_ids)

		return meanNDCG


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""

		avgPrecision = -1

		#Fill in code here
		n_relevant = len(true_doc_IDs)
		found_rel = 0 #counts number of relevant docs found
		is_rel = [] #sores precision of all relevant docs found

		for i in range(k):
			for j in range(n_relevant):
				if(query_doc_IDs_ordered[i] == true_doc_IDs[j][0]):
					found_rel = found_rel+1
					is_rel.append(found_rel/(i+1))
					break;

		
		if found_rel >0:	

			avgPrecision = np.sum(is_rel)/found_rel
		else:
			avgPrecision = 0 #if no relevant doc is found

		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""

		meanAveragePrecision = -1

		#Fill in code here
		meanAveragePrecision = 0
		quer_order, true_ids = extract_true(q_rels)
		for j in range(len(query_ids)):
			for l in range(len(quer_order)):
				if(quer_order[l] == query_ids[j]):
					meanAveragePrecision = meanAveragePrecision + self.queryAveragePrecision(doc_IDs_ordered[j],query_ids[j],true_ids[l], k)
					break;
				
		meanAveragePrecision = meanAveragePrecision/len(query_ids)

		return meanAveragePrecision

