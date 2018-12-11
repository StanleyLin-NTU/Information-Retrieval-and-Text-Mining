import nltk
from nltk.corpus import stopwords
import string
import math

def tokenize(text):
	tokens=list() #list for storing lower-cased tokens 
	tmp=''
	#Start Tokenizing & Lowercasing
	symbols=[" ",",",".","\n","\t","'",'"',":","!","?",">","<","@",";","(",")","[","]","{","}","/","$","#","%","^","&","*","+","=","~","`",'_','-','0','1','2','3','4','5','6','7','8','9']
	for ch in text:
		if ch in symbols:
			if tmp!='':
				tmp=tmp.lower()
				tokens.append(tmp)
				tmp=''
		else:
			tmp=tmp+ch
	#Finish

	new_tokens=list() #list for tokens after stemming by Porter's Algorithm
	#Start Stemming
	porter=nltk.PorterStemmer() #Store the stemmer
	for token in tokens:
		token=porter.stem(token)
		new_tokens.append(token)
	#Finsh Stemming

	stop = set(stopwords.words('english')) #the stoplist of nltk.corpus
	result=list() #store the result

	#Start filtering by stoplist
	for new_token in new_tokens:
		if new_token not in stop:
			result.append(new_token)
	#Finish filtering by stoplist

	return(result)

#main 

token_list=list() #store the tokens of each document
df_dict=dict() #store the df of term

output=open("dictionary.txt",'w') #dictionary

#start calculating df
for index in range(1,1096):
	added=list() #term added df in this document,prevent adding df two times for a term in one document
	
	file_name=str(index)+".txt" #x.txt
	file=open(file_name,'r')
	text_list=file.read()  #read file as string

	tokens=tokenize(text_list) #tokenize
	token_list.append(tokens) #record the tokens for document x, for further use (tf) 

	#Calculate Df
	for term in tokens:
		if(term not in df_dict.keys()): #If term doesn't exist, it is added to dict and df=1
			df_dict[term]=1
			added.append(term)
		else:							#Else, df=df+1 for the term
			if(term not in added): 		#Check whether the term is added df in this doc
				df_dict[term]=df_dict[term]+1
				added.append(term)
	#finish calculate df

#t_index is the term index
t_index=1 

#Outout to "dictionary.txt"
for key in sorted(df_dict):
	string= str(t_index)+"\t"+str(key)+"\t"+str(df_dict[key])
	output.write(string)
	output.write("\n")
	t_index=t_index+1

#change df to idf 
for key in df_dict.keys():
	df_dict[key]=math.log10(float(1095/df_dict[key]))

#Start Calculate tf_idf
for index in range(1,1096):
	tf_index=0 #t index
	index_list=list() #store tf_index of terms for doc x 
	tf_idf_list=list() #store tf_idf of terms for doc x
	num_term=0 #store the number of terms for doc x
	length=0.0

	#output file
	file_name=str(index)+".txt" 
	output_file=open("Vectors/"+file_name,'w') 

	#calculte tf,tf-idf,number of term
	for key in sorted(df_dict.keys()):
		tf_index=tf_index+1 
		if key in token_list[index-1]:
			num_term=num_term+1 #number of term
			num=token_list[index-1].count(key) #tf
			tf_idf=num*df_dict[key] #tf-idf
			tf_idf_list.append(tf_idf) #append it to list
			index_list.append(tf_index) #append the tf_index to list
			length=length+(tf_idf)**2

	output_file.write(str(num_term)) #output the number of term in first line
	output_file.write("\n")

	tf_idf_list=[x/math.sqrt(length) for x in tf_idf_list] #normalize to unit vector

	#output the term index and tf-idf value of term
	for i in range(0,len(tf_idf_list)):		
		output_file.write(str(index_list[i])+"\t"+str(tf_idf_list[i]))
		output_file.write("\n")