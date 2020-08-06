import numpy as np
import csv
from numpy.random import randn
from numpy.random import seed
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import statistics
import matplotlib.pyplot as plt

a1 = []
c1 = []
with open('/home/umangi/Docs/Semester8/NLP/project/brown_lsa/elsa_wordsim.csv', 'r') as sim1:
    reader = csv.reader(sim1)
    for row in reader:
    	a1.append(float(row[2]))
    	c1.append(float(row[3]))

a2 = []
c2 = []
with open('/home/umangi/Docs/Semester8/NLP/project/brown_lsa/elsa_simeval.csv', 'r') as sim1:
    reader = csv.reader(sim1)
    for row in reader:
    	a2.append(float(row[2]))
    	c2.append(float(row[3]))

print(len(c1) , len(a1), len(c2), len(a2))
print(c1[0], a1[0], c2[0], a2[0])
min1 = min(c1)
max1 = max(c1)
min2 = min(c2)
max2 = max(c2)
r1 = max1-min1
r2 = max2-min2
for i in range(len(c1)):
	c1[i] = (c1[i]-min1)/r1
	a1[i] = a1[i]/10

for i in range(len(c2)):
	c2[i] = (c2[i]-min2)/r2
	a2[i] = a2[i]/10

mc1 = statistics.mean(c1)
mc2 = statistics.mean(c2)
ma1 = statistics.mean(a1)
ma2 = statistics.mean(a2)
a1 = [x - ma1 for x in a1] 
a2 = [x - ma2 for x in a2] 
c1 = [x - mc1 for x in c1] 
c2 = [x - mc2 for x in c2]  
for i in range(len(a1)):
    print(i,a1[i], c1[i])
mc1 = statistics.mean(c1)
mc2 = statistics.mean(c2)
ma1 = statistics.mean(a1)
ma2 = statistics.mean(a2)
print(mc1,mc2,ma1, ma2)
# plt.plot(a1, label = 'Wordsim', marker = 'o')
# plt.plot(c1, label = 'LSA- Wordsim', marker = 'o')
plt.scatter(a1,c1, marker = 'o')
plt.xlim([-0.2,0.2])
plt.ylim([-0.2,0.2])
plt.title('Word Pair Similarity Distribution: Wordsim')
plt.legend(loc = 'top right')
plt.show()
# plt.plot(a2, label = 'SimLex', marker = 'o')
# plt.plot(c2, label = 'LSA- SimLex', marker = 'o')
plt.scatter(a2,c2, marker = 'o')
plt.xlim([-0.2,0.2])
plt.ylim([-0.2,0.2])
plt.title('Word Pair Similarity Distribution: SimLex')
plt.legend(loc = 'top right')
plt.show()



corr1, _ = pearsonr(c1, a1)
corr2, _ = pearsonr(c2, a2)
print('Pearsons correlation wordsim: %.3f' % corr1)
print('Pearsons correlation: %.3f' % corr2)

corr1, _ = spearmanr(c1, a1)
corr2, _ = spearmanr(c2, a2)
print('Spearmans correlation simlex: %.3f' % corr1)
print('Spearmans correlation: %.3f' % corr2)
# for i in range(len(a2)):
# 	print(a2[i],c2[i])