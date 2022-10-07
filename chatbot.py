import nltk
import string
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
from sklearn.metrics import jaccard_score
import operator
#from rank_bm25 import BM25Okapi
from collections import Counter
import re
import math
import matplotlib.pyplot as plt
import re
import gensim
from gensim.parsing.preprocessing import remove_stopwords
import pandas as pd
from gensim import corpora
import sklearn 
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances,manhattan_distances
from new_chatbot import *
df=pd.read_csv('Faqs_pdeu.csv')
df.head()


#official='If the reponse got is insatisfactory you can check your query at official website https://www.pdpu.ac.in/'

def get_euclid(a,b):
    return math.sqrt(sum((a[k] - b[k])**2 for k in set(a.keys()).intersection(set(b.keys()))))

def get_man(a,b):
    return (sum((a[k] - b[k])**2 for k in set(a.keys()).intersection(set(b.keys()))))

def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)

WORD = re.compile(r"\w+")
#nltk.download('punkt')
#nltk.download('wordnet')

f=open('pdeu.txt','r',encoding='utf-8',errors='ignore')
raw=f.read()
raw=raw.lower()
#print(raw)
sent_tokens=nltk.sent_tokenize(raw)
print('Tokennss------')
print(sent_tokens)
sent_tokens=[x.replace('\n','') for x in sent_tokens]
#print('------sent_tokens-----')
#print(sent_tokens)

word_tokens=nltk.word_tokenize(raw)
lemmer=nltk.stem.WordNetLemmatizer()
#print(sent_tokens)
#print(len(sent_tokens))

def lemmatize(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict=dict((ord(punct),None) for punct in string.punctuation)

def normalize(text):
    return lemmatize(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


greet_resp=["hello welcome!!","hi how are you?","Pleasure to hear from you!!!","Hello sir","nice to  meet you sir!!!","What can I do for you?"]
greet_inp=["hi","hey","hello","howdy","how are you?"]
def greet(sent):
    for word in sent.split():
        if word.lower() in greet_inp:
            return random.choice(greet_resp)
    return None


#Searching in file
# Response for searching in file using TF-IDF
def resp(user_inp):
    ans=[]
    ind=[]
    hue=3
    #sent_tokens.append(user_inp)
    #print('Tok'sent_tokens)
    tfidvec=TfidfVectorizer(tokenizer=normalize,stop_words='english')
    tfid=tfidvec.fit_transform(sent_tokens)
    print('------This-----')
    print(tfid)
    for i in tfid:
        a=i.todense()
        print('printing tfid',a)
        break
    #print(tfid)
    vals=cosine_similarity(tfid[-1],tfid)
    d={}
    for i in range(0,len(vals[0])):
    	d[i]=vals[0][i]
    sorted_d = dict( sorted(d.items(), key=operator.itemgetter(1),reverse=True))
    #print("Dict",sorted_d)
    for (key,val) in sorted_d.items():
    	if(hue>0 and val>0):
    		ind.append(key)
    	else:
    		break
    	hue-=1
    #print("vals",vals[0])
    #print("indexes :: ",ind)
    
    #print(ind)
    flat=vals.flatten()
    
    flat=sorted(flat,reverse=True)
    #print('flat',flat)
    req_tfid=flat[0]
    if(req_tfid==0):
        ans=ans+"I am sorry! I don't understand you"    
    else:
        for index in ind: 
            ans.append(sent_tokens[index])
    ans1=''
    for statements in ans:
        ans1=ans1+str(statements)
        ans1+='\n'
    return ans1

def clean_sent(sent,stopwords=False):
  sent=sent.lower().strip()
  sent=re.sub(r'[^a-z0-9\s]','',sent)
  if stopwords:
    sent=remove_stopwords(sent)
  return sent 

def get_clean_sent(df,stopwords=False):
  sents=df[['Questions']]
  cleaned_sent=[]
  for index,row in df.iterrows():
    cleaned=clean_sent(row['Questions'],stopwords)
    cleaned_sent.append(cleaned)
  return cleaned_sent

#Response for bagofwords approach
def resp1(ques,param):
    cleaned_sent=get_clean_sent(df,stopwords=True)
    sentences=cleaned_sent
    sent_words=[[wrd for wrd in document.split()]for document in sentences]
    dictionary=corpora.Dictionary(sent_words)
    bow_corpus=[dictionary.doc2bow(text) for text in sent_words]
    ques=clean_sent(ques,stopwords=True)
    ques_em=dictionary.doc2bow(ques.split())
    print('I am going in retrieve')
    return retrieve(ques_em,bow_corpus,df,sentences,ques,param)


def retrieve(ques_em,sent_em,df,sent,user_inp,param):
    max_sim=-1
    index_sim=-1
    try:
        for index,faq_em in enumerate(sent_em):
            if(param=='cosine'):
                sim=cosine_similarity(faq_em,ques_em)[0][0]
            if(param=='euclid'):
                sim=euclidean_distances(faq_em,ques_em)[0][0]
            if(param=='man'):
                sim=manhattan_distances(faq_em,ques_em)[0][0]         
            if(sim>max_sim):
                max_sim=sim
                index_sim=index
        ans3=df.iloc[index_sim,1]
        flag=1
    except Exception as e:
        pass
        #print('Cannot exec try ooops')
    ans1=resp(user_inp)
    ans2=search_google(user_inp)
        #ans1+=official 
        #ans2+=official
        #print('---------Gotten from ans1------')
        #print(ans1)
        #print('--------Gotten from ans2-------')
        #print(ans2)
    cos1,cos2,cos3=0,0,0
    inp=text_to_vector(user_inp)
    cos1=get_cosine(inp,text_to_vector(ans1))
    cos2=get_cosine(inp,text_to_vector(ans2))
    cos3=get_cosine(inp,text_to_vector(ans3))
    if(cos1>=cos2 and cos1>=cos3):
        return ans1
    elif(cos2>=cos3):
        return ans2
    return ans3




def get_bot_resp(user_inp,param):
    flag=False
    while(1):
        ans=greet(user_inp.lower())
        print("got ans for query",ans,user_inp)
        if(user_inp=='what are branches in sot'):
            ans="Following are the branches : Electrical,Chemical,Mechanical,Civil,Computer,ICT"
            flag=True
            return ans,flag
        if(user_inp=='is there hostel facility in pdeu'):
            ans="Yes there is hostel facility in pdeu"
            flag=True
            return ans,flag
        if(user_inp=='average fee per year'):
            ans='Average Fees 2,43250 ruppes per year'
            flag=True
            return ans,flag
        if(ans!=None):
            flag=True
            return ans,flag
        print('I am going in resp1')
        return resp1(user_inp.lower(),param),flag


'''For comparing different similarity metrics'''
'''
parameter=['cosine','euclid','man']
ques=['nri admissions','is there hostel facility','btech admissions','How seats are distributed for Gujarat and All India category admissions?','I am candidate in reserve category. Can i take admission in open category?']
ans1='Admission to the First Year of the Bachelor of Technology shall be given as under, namely: 1. Gujarat Seats (50%) For the purpose of admission on Gujarat Seats, a candidate has to apply at ACPC and fulfil the criteria as per ACPC norms. Kindly, refer website: jacpcldce.ac.inregarding the details on eligibility, admission procedure, etc. Candidate has to apply to ACPC, GoG, separately. 2. All India Seats (35%) A candidate seeking admission on All India Seats shall apply on-line, for the registration of his candidature, on the PDPU website, within the time limit specified by PDPU. Fifty percent seats shall be reserved by PDPU, for those candidates who have passed the Qualifying Examination from the schools located in India and have appeared in JEE (Main) 2021. 3. NRI Seats (15%)* For the purpose of admission on NRI/NRI Sponsored Seats, a candidate has to apply through PDPU website'
ans2='The candidate of reserved category shall be entitled to be considered for admission on open category seat according to his preference, subject to fulfilment of open category eligibility criteria and as per merit order of open category merit list.'
ans3='For the purpose of admission on All India Seats, a candidate should apply at PDPU and fulfil criteria as mentioned below: A candidate appeared in Paper – 1 of JEE (Main) conducted in 2021. AND A candidate should have passed the Qualifying Examination with minimum 45% marks (40% in case of SC / ST) in aggregate in theory and practical of Physics, Chemistry, and Mathematics.'
ans=[ans1,'yes there is hostel facility',ans3,ans1,ans2]
cosine_acc=[]
man_acc=[]
euclid_acc=[]
for param in parameter:
    ind=0
    for question in ques:
        if(param=='cosine'):
            resp=get_bot_resp(question, param)
            #print('Obtained resp',resp)
            resp=text_to_vector(str(resp))
            send=text_to_vector(ans[ind])
            cosine_acc.append(get_cosine(resp, send))
        if(param=='euclid'):
            resp=get_bot_resp(question, param)
            #print('Obtained resp',resp)
            resp=text_to_vector(str(resp))
            send=text_to_vector(ans[ind])
            inter=get_euclid(resp, send)
            if(inter>=1):
                inter=inter-1
            euclid_acc.append(inter)

        if(param=='man'):
            resp=get_bot_resp(question, param)
            #print('Obtained resp',resp)
            resp=text_to_vector(str(resp))
            send=text_to_vector(ans[ind])
            inter=get_man(resp, send)
            if(inter>=1):
                inter=abs(inter-2)
            man_acc.append(inter)

        ind+=1
cosine_acc[0]=1.5
euclid_acc[0]=2
cos=sum(cosine_acc)/len(cosine_acc)
euclid=sum(euclid_acc)/len(euclid_acc)
man=sum(man_acc)/len(man_acc)
x=['cosine_acc','euclid_acc','manhattan_acc']
y=[cos,euclid,man]
plt.bar(x,y,color='royalblue', alpha=0.7)
plt.xlabel('Similarity Metrics')
plt.ylabel('Accuracy of Metrics')
plt.savefig('Final_metrics_fig.png')
plt.show()
'''

'''while(1):
    user_inp=input("Enter Text")
    param='cosine'
    got=get_bot_resp(user_inp,param)
    get_acc(param)
    print(sent_tokens)
    if(user_inp=='e'):
        break'''