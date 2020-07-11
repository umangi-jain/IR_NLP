#This function takes in qrels and returns true_ids which is a list of list of tuples 
#(containing id and relevance) for queries in quer_orde respectively.
def extract_true(qrels):
	id_then = -1
	id_now = -1
	true_ids = []
	temp_ids = []#list of tuples for a particular query

	quer_order = []
	for i in range(len(qrels)):
		id_now = int(qrels[i]["query_num"])
		if i ==0: #only for first query
			id_then = id_now
		if(id_now != id_then): #new query

			quer_order.append(id_then)
			true_ids.append(temp_ids)
			temp_ids = []
			temp_ids.append((int(qrels[i]["id"]), qrels[i]["position"])) 
			
		else:
			temp_ids.append((int(qrels[i]["id"]), qrels[i]["position"])) 
			#temp_ids.append(qrels[i]["id"])
		id_then = id_now
	quer_order.append(id_now)
			
	temp_ids.append((int(qrels[i]["id"]), qrels[i]["position"])) #append for last query
	true_ids.append(temp_ids)
	return (quer_order, true_ids)