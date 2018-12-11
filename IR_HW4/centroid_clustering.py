import numpy as np 
import math
import heapq

num_of_cluster=1095
#initialize: each point is a cluster (bottom-up)
original_vec= np.empty((1095, 12347), float) #vector 

#store the normalized vectors
for x in range(1,1096):
  d1="normalized_"+str(x)+".txt"
  f1= open(d1,'r') 
  vec1=f1.read().splitlines()
  vec1=[float(i) for i in vec1]
  original_vec[x-1]=vec1

Name_Vector = [[x] for x in range(1,1096)] #store the cluster members
Num_of_doc=[1]*1095 #store for cluster member count
heap=[] #heap for stroing cosine similarity

#create heap by cosine similarity between doc (initialization)
for key_x in range(0,1095):
  for key_y in range(key_x+1,1095):
   if(key_x!=key_y):
      value=(-1)*np.dot(original_vec[key_x],original_vec[key_y])
      heapq.heappush(heap,(value,key_x+1,key_y+1)) 

delete=list() #Record which docs have been merged

while (num_of_cluster>8): 
  heap_max=heapq.heappop(heap) #get the max cosine_similarity value
  max_one=(heap_max[1]) #doc_one
  max_two=(heap_max[2]) #doc_two
  delete.append(max_two) #delete doc two

  #update the new vector [max_one-1]
  original_vec[max_one-1]=(original_vec[max_one-1]*Num_of_doc[max_one-1]+original_vec[max_two-1]*Num_of_doc[max_two-1])/(Num_of_doc[max_one-1]+Num_of_doc[max_two-1])
  Name_Vector[max_one-1]=Name_Vector[max_one-1]+Name_Vector[max_two-1] #Update the cluster members

  #delete the items contain doc one and two by creating a new heap
  heap = [(x,d1,d2) for (x,d1,d2) in heap if d1!=max_one and d1!=max_two and d2!=max_one and d2!=max_two]
  heapq.heapify(heap) #fulfill heap feature

  #push the new points (merged points to heap)
  for i in [x for x in range(1,1096) if x not in delete and x!=max_one]:
    value=(-1)*np.dot(original_vec[max_one-1],original_vec[i-1])
    heapq.heappush(heap,(value,max_one,i))

  #update number of clusters
  Num_of_doc[max_one-1]=Num_of_doc[max_one-1]+Num_of_doc[max_two-1]
  Num_of_doc[max_two-1]=0  

  #finish, update num_of_cluster
  num_of_cluster=num_of_cluster-1

  #output if num_of_cluster is 8, 13 or 20
  if(num_of_cluster==8 or num_of_cluster==13 or num_of_cluster==20):
    Real_Name_Vector=list() #for sorting
    f_name=str(num_of_cluster)+".txt"
    file_cluster=open(f_name,'w')
    for i in range(1,1096):
      if(i not in delete):  
        Name_Vector[i-1].sort()
        Real_Name_Vector.append(Name_Vector[i-1])
    Real_Name_Vector.sort(key=lambda x: x[0])
    for i in range(0,len(Real_Name_Vector)): 
      for x in Real_Name_Vector[i]:
        file_cluster.write(str(x))  
        file_cluster.write("\n")
      file_cluster.write("\n")  