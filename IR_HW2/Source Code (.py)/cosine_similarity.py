import numpy as np 
import math

#the function which calculates cosine similarity of two documents
def cosine(d1,d2): #d1,d2: the path of document 1 and 2
	f1= open(d1,'r') 
	f2= open(d2,'r') 
	next(f1) #we don't need the first line (number of terms of document)
	next(f2) #we don't need the first line (number of terms of document)
	vec1=np.array([0.0]*12347) #vec1, a np array for d1
	vec2=np.array([0.0]*12347) #vec2, a np array for d2

	#read value of d1 and store in vec1
	for x in f1.readlines():
  		tmp=x.strip().split("\t")
  		vec1[int(tmp[0])-1]=float(tmp[1]) 
  		#tmp[0] is term index (1-12347), so the index in array is i-1 (0-12346)
  		#tmp[1] is the tf-idf value of the term

	#read value of d2 and store in vec2
	for y in f2.readlines():
  		tmp=y.strip().split("\t")
  		vec2[int(tmp[0])-1]=float(tmp[1])
  		#tmp[0] is term index (1-12347), so the index in array is i-1 (0-12346)
  		#tmp[1] is the tf-idf value of the term

	inner_product=np.inner(vec1,vec2) #inner product of [vec1/|vec1|], [vec2/|vec2|]
	
	return(inner_product)


# main 
document1="1.txt" #change to which document you want
document2="2.txt" #change to which document you want
value=cosine(document1,document2) #call the function
print("The cosine similarity of "+'"'+str(document1)+'"' +" and "+ '"'+str(document2)+'"'+ " is: "+str(value))