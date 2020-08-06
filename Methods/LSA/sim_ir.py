from util import *
import numpy as np
import nltk
nltk.download('brown')
from nltk.corpus import brown
from nltk.corpus import stopwords 
import csv
# Add your import statements here


class InformationRetrieval():

	def __init__(self):
		self.index = None	# Stores the inverted index representation for the documents, which is built in the buildIndex() funciton
		self.idf_of_terms = None
		self.N = 0 # Stores the total number of documents
		self.index_terms = {}
		self.terms = set([]) # stores the terms present in documents and in queries, which would form the terms that form the dimentions in the vector space
		self.docIDs = [] # stores the document ids of all the documents. It is initialised from the docIDs list passed in the buildIndex function and used in other class functions.
		self.words = []
		self.bterms = []
		self.bdocs = []
	def add_brown(self):
		self.bwords = []
		stop_words = set(stopwords.words('english')) 
		words = []
		# words = brown.words(categories=['news'])
		# print("total brown words = ", len(words))
		# for i in words:
		# 	if i in stop_words or len(i)<3:
		# 		pass
		# 	else:
		# 		self.bterms.append(i)
		# print("filtred brown words = ", len(self.bterms))
		words = set([])
		self.bdocs = brown.paras(categories=['news']) #15667 list of list of list of strings
		print("brown docs ", len(self.bdocs))
		self.bdocs = self.bdocs[:2000]
		for doci in self.bdocs:
			for senti in doci:
				for wordi in senti:
					words.add(wordi.lower())
		print("Unfiltered words ", len(words))

		for i in words:
			if i in stop_words or len(i)<3:
				pass
			else:
				self.bterms.append(i)
		print("filtred brown words = ", len(self.bterms))

		# self.bterms = self.bterms[:5]

	def buildmatrix(self, docs, queries, docIDs, given ):
		N = len(docs)
		self.td = N
		self.docIDs = docIDs

		self.index = {}
		for doc in docs:
			for sentence in doc:
				for word in sentence:
					self.terms.add(word.lower())
		print("Cranfield words = ", len(self.terms))
		#######
		self.add_brown()
		# print(len(self.bterms))
		# for i in self.bterms:
		# 	self.terms.add(i.lower())
		print(" ALL combined words = ", len(self.terms))

		print("Begin combining doccs")
		docs.extend(self.bdocs)
		######
		self.ld = len(docs)
		# print("End combining")
		# print("length of all docs ", len(docs))

		# print("Begin stupidity")
		i =0
		for term in self.terms:
			self.index_terms[term] = i
			i = i+1


		# index_docs = {}
		# for j in range(self.ld):
		# 	index_docs[docIDs[j]] = j
			
		# print("end stupidity")

		print("Begin builidng index")
		ii =0
		for term in self.terms:
			if(ii%1000 == 0):
				print("index for words ", ii)
			ii = ii+1

			self.index[term] = []
			for i in range(len(docs)):
				tf = 0
				for sentence in docs[i]:
					for word in sentence:
						if word.lower() == term:
							tf += 1
				# the entry for a document is included in index[term] only if the term is present atleast once in the in that doc (tf > 0)
				if tf>0:
					self.index[term].append((i,tf))

			# arranging the list of tuples in decreasing order of tf of the term in the documents
			# index[term] = sorted(self.index[term], key = lambda a: a[1], reverse=True)
		print("End building")
		print("Begin idf")
		self.idf_of_terms = {}
		for term in self.terms:
			if term in self.index.keys(): # term present in atleast one of the documents
				n = len(self.index[term]) # n is the number of documents in which that term is present.
				if(n ==0):
					idf = 0
				else:
					idf = np.log10(N/n)
			else:
				idf = 0

			self.idf_of_terms[term] = idf

		print("End idf")
		
		M = len(self.terms)
		tdm = np.zeros([M, self.ld])
		#td matrix
		print("tdm matrix")
		i = 0
		for term in self.terms:
			if(i%1000 == 0):
				print("index for words ", i)
			i = i+1
			for jj in self.index[term]:
				tdm[self.index_terms[term]][jj[0]] = jj[1]*self.idf_of_terms[term]

		print("END of tdm matrix")
		x = np.linalg.norm(tdm, axis=0)  
		print("NORMALIZATION ", len(tdm[0]))
		for ip in range(len(tdm[0])):
			if(x[ip]) == 0:
				print("ALERT ", ip)
			else:
				tdm[:,ip] = tdm[:,ip]/x[ip]
		# for k in range(N):
		# 	doc = docs[k]
		# 	for sentence in doc:
		# 		for word in sentence:
		# 			if word.lower() in self.terms:
		# 				# if (k == 50):
		# 				# 	print(word.lower(),index_terms[word.lower()],[index_docs[docIDs[k]]])
		# 				tdm[self.index_terms[word.lower()]][index_docs[docIDs[k]]] += 1*self.idf_of_terms[word.lower()]

		print(np.shape(tdm))
		# print("end")
		# f = open("demofile1.txt", "a")
		# f.write(str(self.index_terms))
		# f.close()
		# f = open("demofile2.txt", "a")
		# np.savetxt('demofile2.txt', tdm[:, 50])
		# print(docs[50])
		# f.close()

		
		
		t, s, dh = np.linalg.svd(tdm, full_matrices=False)
		print(np.shape(t), np.shape(s), np.shape(dh))
		k = 1500#self.ld #given #self.ld #concepts
		t = t[:,:k]
		s = np.diag(s[:k])
		dh = dh[:k,:]
		print(np.shape(t), np.shape(s), np.shape(dh))
		temp = np.matmul(s,dh)
		tdmn = np.matmul(t,temp)
		print(np.shape(tdmn))

		a = np.matmul(s,s)
		b = np.matmul(t,a)
		c = np.matmul(b, np.transpose(t))
		print("measuring word similarity begins")

		simeval_pairs = []
		simeval_scores = []
		this_scores = []
		with open('/home/umangi/Docs/Semester8/NLP/project/wordsim evaluation/simlex_common.csv', 'r') as sim1:
		    reader = csv.reader(sim1)
		    for row in reader:
		    	simeval_pairs.append([row[0], row[1]])
		    	simeval_scores.append(row[2])

		for rr in range(len(simeval_scores)):
			x = self.index_terms[simeval_pairs[rr][0]]
			y = self.index_terms[simeval_pairs[rr][1]]
		
			this_scores.append(c[x][y])
		print("simlex ", len(simeval_pairs))  
		with open('elsa_simeval.csv', mode='w') as employee_file:
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
			x = self.index_terms[simeval_pairs[rr][0]]
			y = self.index_terms[simeval_pairs[rr][1]]
		
			this_scores.append(c[x][y])

		with open('elsa_wordsim.csv', mode='w') as employee_file:
		    employee_writer = csv.writer(employee_file, delimiter=',')
		    for pl in range(len(simeval_pairs)):
		    	employee_writer.writerow([simeval_pairs[pl][0], simeval_pairs[pl][1], str(simeval_scores[pl]), str(this_scores[pl])])


		return tdmn, t,s,dh,k


	





	

	def rank1(self, tdmn, t,s,dh, queries, k):
		M = np.shape(tdmn)[0] #terms
		N = np.shape(tdmn)[1] #docs all
		si = np.linalg.inv(s)
		ss = np.matmul(s,s)

		doc_IDs_ordered = []
		for i in range(len(queries)):
			query = queries[i]
			
			xd = np.zeros([M,1],dtype='float')










			for sentence in query:
				for word in sentence:
					if (word.lower() in self.terms):
						xd[self.index_terms[word.lower()],0] += 1*self.idf_of_terms[word.lower()]
						# if( i==1):
						# 	print(word.lower(),self.index_terms[word.lower()])
			
			temp = np.matmul(np.transpose(xd), t)
			qd = np.matmul(temp,si)#pseudo doc
			x = np.linalg.norm(dh, axis=0)  
			# print("NORMALIZATION ", len(dh[0]))
			for ip in range(len(dh[0])):
				if(x[ip]) == 0:
					print("ALERT ", ip)
				else:
					dh[:,ip] = dh[:,ip]/x[ip]
				

			valt = np.matmul(qd,ss)
			val = np.matmul(valt, dh)
			# print(np.shape(qd), np.shape(tdmn))
			# val = np.matmul(np.transpose(xd), tdmn)
			val = val[0,:]
			# print("HERE ", np.shape(val))
			doc_IDs_ordered_it = np.argsort(val)
			
			doc_IDs_ordered_i = doc_IDs_ordered_it[::-1]


			# f = open("demofile2.txt", "a")
			# np.savetxt('demofile2.txt', val)
			# gogli = np.sort(val)
			# testa = np.argmax(val)
			# testb = val[testa]
			# print("1")
			# print(gogli[-1],testb)
			# print("2")
			# print(testa,doc_IDs_ordered_i[0], doc_IDs_ordered_i[-1] )
			# print("END")
			# print("doq")
			# print(gogli[-5:])
			# np.savetxt('demofile3.txt', doc_IDs_ordered_i)
			
			# f.close()
			doc_IDs_ordered.append(doc_IDs_ordered_i)

			
		print(np.shape(doc_IDs_ordered))
		# f = open("demofile2.txt", "a")
		# np.savetxt('demofile2.txt', doc_IDs_ordered[1])
		
		# f.close()
		return doc_IDs_ordered
			
			




