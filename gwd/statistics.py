import matplotlib.pyplot as plt
from scipy import *
import sys

"""
Taking the total energy of i's iteration
shows its tendency or convergence over steps
"""

myfile = open('data','r') #open the data file
data = myfile.readlines() #taking each line as a single list

Eitt = []
xitt = []

print len(data)
for i,line in enumerate(data): #for each line
	words = line.split() #get each word
	Nword = len(words) #
	if Nword > 3 and words[3] =='itt':
		itt = int(words[5])
		R = float(words[0])
		e = float(words[1])
		
		Eitt.append(e)
		xitt.append(itt)

plt.plot(xitt,Eitt)
plt.show()
