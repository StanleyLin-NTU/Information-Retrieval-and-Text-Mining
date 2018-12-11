import numpy as np

for x in range(1,1096):
  file_name="normalized_"+str(x)+".txt"
  output_f=open(file_name,'w')
  #print(x)
  d1="Original_"+str(x)+".txt"
  f1= open(d1,'r') 
  next(f1) #we don't need the first line (number of terms of document)
  vec1=np.array([0.0]*12347) #vec1, a np array for d1

  #read value of d1 and store in vec1
  for x in f1.readlines():
      tmp=x.strip().split("\t")
      vec1[int(tmp[0])-1]=float(tmp[1]) 
      #tmp[0] is term index (1-12347), so the index in array is i-1 (0-12346)
      #tmp[1] is the tf-idf value of the term
  for a in vec1:
    output_f.write(str(a))
    output_f.write("\n")    