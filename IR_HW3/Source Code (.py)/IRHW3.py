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

#Feature Selection Starts #

f1=open("class.txt","r")
classes=f1.read().splitlines()

for i in range(0,13):
	tmp=classes[i].strip().split(" ")
	tmp.pop(0)
	classes[i]=tmp #the class docs for class i+1

class_token=list() #list of list of list (class-docs-tokens)
terms=dict() #store all the terms and the LLR for classes
train_docs=list() #train docs

for lists in classes:
	token_list=list() #store the tokens of each document
	for items in lists:
		train_docs.append(int(items))
		file_name=str(items)+".txt" #x.txt
		file=open(file_name,'r')
		text_list=file.read()  #read file as string
		tokens=tokenize(text_list) #tokenize
		for tok in tokens:
			if(tok not in terms.keys()): #terms of all classes
				terms[tok]=list()
		token_list.append(tokens) #record the tokens of docs in class X

	class_token.append(token_list) #record the tokens of docs of each class to class_token

#Calculating log likelihood ratio feature selection
for item in terms.keys():
	
	for x in range(0,len(class_token)): #run through each class
		on_appear=0 #on topic, present
		off_appear=0 #off topic, present
		on_absent=0 #on topic, absent
		off_absent=0 #off topic, absent

		#check on topic
		for doc_token in class_token[x]:
			if(item in doc_token): #present
				on_appear=on_appear+1
			else:
				on_absent=on_absent+1

		#check off topic		
		for y in range(0,len(class_token)):
			if(y!=x):
				for doc_token in class_token[y]:
					if(item in doc_token): #present
						off_appear=off_appear+1
					else:
						off_absent=off_absent+1
				
		conf=[on_appear,on_absent,off_appear,off_absent]
		pt= (conf[0]+conf[2])/sum(conf) #pt for Hypothesis 1
		p1= (conf[0])/(conf[0]+conf[1]) #p1 for Hypothesis 2
		p2= (conf[2])/(conf[2]+conf[3]) #p2 for Hypothesis 2
		LR=((pt**conf[0])*((1-pt)**conf[1])*(pt**conf[2])*((1-pt)**conf[3]))/((p1**conf[0])*((1-p1)**conf[1])*(p2**conf[2])*((1-p2)**conf[3]))
		LLR=(-2)*math.log2(LR) #store LR and LLR
		terms[item].append(LLR) #record the LLR for class x

#finish saving all matrices for terms



dict_LLR=dict() #store the average LLR for term in classes
filter_term=dict() #500 terms with max LLR

#average the LLRs
for item in terms.keys():
	dict_LLR[item]=sum(terms[item])/len(terms[item])

#Start choosing max 500 terms
count=0
for key in sorted(dict_LLR,key=dict_LLR.get,reverse=True):
	if(count<500):
		count=count+1
		filter_term[key]=list()
#Finish choosing

#Finish feature selection# 

#save new token, ignoring tokens not filtered terms
new_class_token=list() #new tokens for docs , list of list of list (class-docs-tokens)
for c in class_token:
	tmp_class=list()
	for doc_token in c:
		tmp_doc=list()
		for t in doc_token:
			if(t in filter_term.keys()):
				tmp_doc.append(t)
		tmp_class.append(tmp_doc)
	new_class_token.append(tmp_class)

#Start Multi-nominal Naive Bayes Classfication

#Train Start
Voc_size=500 #Term Size
Prob_class=1/13 #Probabilty of each class is 1/13
for term in filter_term.keys():
	for c in new_class_token:
		class_tok_count=0
		term_appear=0
		for doc_tok in c:
			class_tok_count=class_tok_count+len(doc_tok) #record total tokens in class
			for tok in doc_tok:
				if(term==tok):
					term_appear=term_appear+1 #occurence of the term in a class
		prob_of_class=(term_appear+1)/(Voc_size+class_tok_count)
		filter_term[term].append(prob_of_class) #record the prob in filter_term
#Finish training

#Testing Start
file_answer=open("B03705002.txt",'w')
for i in range(1,1096):
	if i not in train_docs: #exclude training
		predicts=[math.log2(Prob_class)]*13
		file_test=str(i)+".txt" #x.txt
		file=open(file_test,'r')
		text_list=file.read()  #read file as string
		tokens=tokenize(text_list) #tokenize
		for word in tokens:
			if(word in filter_term.keys()): #only choose filtered terms
				for x in range(0,13):
					predicts[x]=predicts[x]+math.log2(filter_term[word][x]) #use log to prevent *0

		max_value=max(predicts) 
		class_of_doc=predicts.index(max_value)+1 #get class

		#Output
		write_content=str(i)+"\t"+str(class_of_doc)
		file_answer.write(write_content)
		file_answer.write("\n")

#Finish Multi-nominal Naive Bayes Classfication
