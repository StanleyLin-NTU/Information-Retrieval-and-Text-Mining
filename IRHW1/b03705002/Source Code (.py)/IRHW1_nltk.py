import nltk 
from nltk.corpus import stopwords
#nltk is a special package, which I used for stemming by Potter's Algorithm
#stopwords is a special dataset of nltk.corpus(need to be downloaded additionally)

f1=open("/Users/StanleyLIn/Desktop/paragraph.txt",'r') #Directory need to be changed
#IRHW1.txt is the original document provided by professor
text=f1.read()

tokens=list() #list for storing lower-cased tokens 
tmp=''
#Start Tokenizing & Lowercasing
for ch in text:
	if ch==" " or ch== "," or ch=="." or ch=="\n" or ch=="'":
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

#Output the result as Output.txt,each token is divided by \n
output_file=open("/Users/StanleyLIn/Desktop/Output_nltk.txt",'w') #Directory need to be changed
for ch in result:
	output_file.write(ch)
	output_file.write("\n")
#Finish Outputing