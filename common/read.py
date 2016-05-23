
from scipy import *
import os
which = range(600)
ER = []
S = []
for iR in which:
	if iR%10==0:
		r = iR/100.
		s = "%1.1f" %r
		S.append(s)
		s = "%1.2f" %r
		S.append(s)
	else:
		r = iR/100.
		s = "%1.2f" %r
		S.append(s)
		
#print S

for s in S:
	#if os.path.exists('954'+str(i)+'naivephya/data'):
	#	myfile = open('954'+str(i)+'naivephya/data','r')
	nitt = 0
	res = 0.0
	if os.path.exists(s+'/data'):
		myfile = open(s+'/data','r')
		data = myfile.readlines()
		for i,line in enumerate(data):
			words = line.split()
			Nword = len(words)
			if Nword > 3 and words[3] =='itt':
				itt = int(words[5])
				if itt >20:
					nitt += 1
					R = float(words[0])
					e = float(words[1])
					res+=e
					
	if nitt >0:
		ER.append([R,res/nitt])

ER = array(ER)
print ER
savetxt('ER.dat',ER,delimiter='	', fmt ="%.6f %0.6f")
