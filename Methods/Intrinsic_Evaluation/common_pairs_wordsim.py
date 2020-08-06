from util import *
import numpy as np
import csv
# Add your import statements here


class Commonwords():


	def read_text(self, file_name, file_name2):
	    file_data = []
	    text_file = open(file_name,"r")
	    sim_score = []
	
	    for word in text_file.read().split():
	    	if(word[0]>='A'):
	    		file_data.append(word.lower())
	    	else:
	    		sim_score.append(float(word))

	    text_file.close()
	    pair_data = []
	    temp = []
	    for i in range(len(file_data)):
	    	if i%2 == 0 :
	    		temp.append(file_data[i])
	    	elif i%2 == 1:
	    		temp.append(file_data[i])
	    		pair_data.append(temp)
	    		temp = []
	    text_file.close()
	    ###########################
	    text_file = open(file_name2,"r")
	    
	   
	    for word in text_file.read().split():
	    	if(word[0]>='A'):
	    		file_data.append(word.lower())
	    	else:
	    		sim_score.append(float(word))

	    text_file.close()
	    pair_data = []
	    temp = []
	    for i in range(len(file_data)):
	    	if i%2 == 0 :
	    		temp.append(file_data[i])
	    	elif i%2 == 1:
	    		temp.append(file_data[i])
	    		pair_data.append(temp)
	    		temp = []
	    text_file.close()



	    final_data= []
	    final_scores = []
	    final_data.append(pair_data[0])
	    final_scores.append(sim_score[0])
	    for i in range(1,len(pair_data)):
	    	fe = 0
	    	for j in range( len(final_data)):
	    		if pair_data[i][0] == final_data[j][0] and pair_data[i][1] == final_data[j][1]:
	    			fe = 1
	    			break
	    	if fe == 0:	
	    		final_data.append(pair_data[i])
	    		final_scores.append(sim_score[i])

	    return final_data, final_scores
	    # for word in text_file.read().split():
	    	
	    #     # i = i+1
	    #     if(word[0]>='A'):
	    #     	# if(i%2 == 1):
	    #     	# 	j.append(word)
	    #     	# else:
	    #     	# 	j.append(word)
	    #     	if word.lower() in file_data:
	    #     		foo = 2
	    #     	else:
     #    			file_data.add(word.lower())
     #    			foo = 1
     #    			print("ping")
	    #     		# j = []
	    #     elif foo == 1:
	    #     	sim_score.append(float(word))
	    #     	foo  =0
	    #     	print("pong")

	    

	def read_sim(self, file_name):
	    file_data = []
	    text_file = open(file_name,"r")
	    foo = 0
	    sim_score = []
	    for word in text_file.read().split():
	    	if foo < 10:
	    		pass

	    	else:
		    	if (foo-3)%10 == 0 :
		    		
		    		sim_score.append(float(word))
		    	
		    	if(word[0]>='A'):
		        	if(word == 'A' or word =='a'):
		        		pass
		        	elif(word == 'V' or word =='v'):
		        		pass
		        	elif(word == 'N' or word =='n'):
		        		pass
		        	# if(i%2 == 1):
		        	# 	j.append(word)
		        	# else:
		        	# 	j.append(word)
		        	else:
	        			file_data.append(word.lower())

	    	foo = foo+1	        		# j = []
	       



	    text_file.close()
	    
	    pair_data = []
	    temp = []
	    for i in range(len(file_data)):
	    	if i%2 == 0 :
	    		temp.append(file_data[i])
	    	elif i%2 == 1:
	    		temp.append(file_data[i])
	    		pair_data.append(temp)
	    		temp = []
	    	

	    return pair_data ,sim_score   

	def __init__(self):
		self.index = None	# Stores the inverted index representation for the documents, which is built in the buildIndex() funciton
		self.N = 0 # Stores the total number of documents
		self.index_terms = {}
		self.terms = set([]) # stores the terms present in documents and in queries, which would form the terms that form the dimentions in the vector space
		self.docIDs = [] # stores the document ids of all the documents. It is initialised from the docIDs list passed in the buildIndex function and used in other class functions.

	def buildmatrix(self, docs, queries, docIDs ):
		N = len(docs)
		self.docIDs = docIDs

		# Collecting all the terms in the documents into the set self.terms
		for doc in docs:
			for sentence in doc:
				for word in sentence:
					self.terms.add(word.lower())


		print(len(self.terms))
		the_list = []
		the_scores = []
		a,b = self.read_text('wordsim_relatedness_goldstandard.txt','wordsim_similarity_goldstandard.txt')
		the_list.extend(a)
		the_scores.extend(b)
		# a,b = self.read_text('wordsim_similarity_goldstandard.txt')
		# the_list.extend(a)
		# the_scores.extend(b)
		print("wordsim Input word pairs length")
		print(len(the_list))
		print("wordsim Input word scores length")
		print(len(the_scores))
		
		all_pair = []
		all_scores = []
		
		ii = -1
		for words in the_list:
			# print(words)
			ii = ii+1
			bar = 0
			for terms in self.terms:
				if(words[0] == terms):
					bar = bar+1
				if(words[1] == terms):
					bar = bar+1
			if(bar ==2):
				all_pair.append(words)
				all_scores.append(the_scores[ii])

		print("Print all common pairs")
		print(len(all_pair))
		print(len(all_scores))
		print(all_pair[16], all_scores[16])

		with open('wordsim_common.csv', mode='w') as employee_file:
		    employee_writer = csv.writer(employee_file, delimiter=',')
		    for pl in range(len(all_pair)):
		    	employee_writer.writerow([all_pair[pl][0], all_pair[pl][1], str(all_scores[pl])])

		sim_list ,sim_score = self.read_sim('SimLex-999.txt')
		
		print("Dataset 2 Input word pairs")
		print(len(sim_list), len(sim_score))
		# all_common_list = []
		# for words in sim_list:
		# 	for word in words:
		# 		for terms in self.terms:
		# 			if(word == terms):
		# 				all_common_list.append(word)
		# print("All common words")
		# print(len(all_common_list))
		allc_pair = []
		allc_scores = []
		# print("BEGIN")
		# print(the_list)
		# print("END")
		ii = -1
		for words in sim_list:
			ii = ii+1
			# print(words)
			bar = 0
			for terms in self.terms:
				if(words[0] == terms):
					bar = bar+1
				if(words[1] == terms):
					bar = bar+1
			if(bar ==2):
				allc_pair.append(words)
				allc_scores.append(sim_score[ii])
		print("Print all common pairs")
		print(len(allc_pair) ,len(allc_scores))
		print(allc_pair[21], allc_scores[21])
		with open('simlex_common.csv', mode='w') as employee_file:
		    employee_writer = csv.writer(employee_file, delimiter=',')
		    for pl in range(len(allc_pair)):
		    	employee_writer.writerow([allc_pair[pl][0], allc_pair[pl][1], str(allc_scores[pl])])



