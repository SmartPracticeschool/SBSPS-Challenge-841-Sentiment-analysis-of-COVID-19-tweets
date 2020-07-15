#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 13:51:20 2020

@author: sanjanasrinivasareddy
"""
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

#Model Selection and Validation
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
import nltk
import sklearn
import pandas as pd
df1 = pd.read_csv('file.csv', encoding = 'unicode_escape')
df2 = pd.read_csv('file2.csv', encoding = 'unicode_escape')
df3= pd.read_csv('file3.csv', encoding = 'unicode_escape')
df4= pd.read_csv('file4.csv', encoding = 'unicode_escape')
df5= pd.read_csv('file5.csv', encoding = 'unicode_escape')
df6= pd.read_csv('file6.csv', encoding = 'unicode_escape')
df7= pd.read_csv('file7.csv', encoding = 'unicode_escape')

##########
df8= pd.read_csv('file8.csv', encoding = 'unicode_escape')
df9= pd.read_csv('file9.csv', encoding = 'unicode_escape')
##########



print(len(df1.index))
df1=df1.drop('Id',1)

df2=df2.drop('Id',1)
df2=df2.drop('Unnamed: 0',1)

df3=df3.drop('Id',1)
df3=df3.drop('Unnamed: 0',1)

df4=df4.drop('Id',1)
df4=df4.drop('0',1)

df5=df5.drop('Id',1)
df5=df5.drop('0',1)

df6=df6.drop('Id',1)
df6=df6.drop('Unnamed: 0',1)

df7=df7.drop('Id',1)
df7=df7.drop('Unnamed: 0',1)

##############
df8=df8.drop('Id',1)
df8=df8.drop('900',1)

df9=df9.drop('Id',1)
df9=df9.drop('5900',1)

##############

all_dfs = [df1, df2,df3,df4,df5,df6,df7]
df=pd.concat(all_dfs).reset_index(drop=True)


#######
new_df=[df8,df9]
new_df=pd.concat(new_df).reset_index(drop=True)
serlis=new_df.duplicated(['Tweet']).tolist()
print(serlis.count(True))

new_df=new_df.drop_duplicates(['Tweet'])

new_df=new_df.iloc[1:,:]
new_df=new_df.reset_index(drop=True)

#######

#serlis=df.duplicated().tolist()
serlis=df.duplicated(['Tweet']).tolist()
print(serlis.count(True))

df=df.drop_duplicates(['Tweet'])

df=df.iloc[1:,:]
df=df.reset_index(drop=True)


nltk.download('wordnet')        
def normalization(tweet_list):
        lem = WordNetLemmatizer()
        normalized_tweet = []
        for word in tweet_list:
            normalized_text = lem.lemmatize(word,'v')
            normalized_tweet.append(normalized_text)
        return normalized_tweet

def no_user_alpha(tweet):
    tweet_list = [ele for ele in tweet.split() if ele != 'user']
    clean_tokens = [t for t in tweet_list if re.match(r'[^\W\d]*$', t)]
    clean_s = ' '.join(clean_tokens)
    clean_mess = [word for word in clean_s.split()]
    return clean_mess
import nltk
nltk.download('words')
words = set(nltk.corpus.words.words())
import re
for i in range(len(df)):
    print(i)
    df['Tweet'][i]=" ".join(w for w in nltk.wordpunct_tokenize(df['Tweet'][i]) if w.lower() in words or not w.isalpha()) #remove non english words    
    txt = df.loc[i]["Tweet"]
    txt=re.sub(r'@[A-Z0-9a-z_:]+','',txt)#replace username-tags
    txt=re.sub(r'^[RT]+','',txt)#replace RT-tags
    txt = re.sub('https?://[A-Za-z0-9./]+','',txt)#replace URLs
    txt=re.sub("[^a-zA-Z]", " ",txt)#replace hashtags
    df.at[i,"Tweet"]=txt
    df['Tweet'][i]=no_user_alpha(df['Tweet'][i])
    df['Tweet'][i]=normalization(df['Tweet'][i])
    if(df['Sentiment'][i]>=0):
        df['Sentiment'][i]=1
    else:
        df['Sentiment'][i]=0
    
#delete_row = df[df.iloc[:,2]==" "].index
#df = df.drop(delete_row)    
        
###############
df=new_df
for i in range(len(df)):
    print(i)
    df['Tweet'][i]=" ".join(w for w in nltk.wordpunct_tokenize(df['Tweet'][i]) if w.lower() in words or not w.isalpha()) #remove non english words    
    txt = df.loc[i]["Tweet"]
    txt=re.sub(r'@[A-Z0-9a-z_:]+','',txt)#replace username-tags
    txt=re.sub(r'^[RT]+','',txt)#replace RT-tags
    txt = re.sub('https?://[A-Za-z0-9./]+','',txt)#replace URLs
    txt=re.sub("[^a-zA-Z]", " ",txt)#replace hashtags
    df.at[i,"Tweet"]=txt
    df['Tweet'][i]=no_user_alpha(df['Tweet'][i])
    df['Tweet'][i]=normalization(df['Tweet'][i])
    if(df['Sentiment'][i]>=0):
        df['Sentiment'][i]=1
    else:
        df['Sentiment'][i]=0
        
#################
df_new=df        
df_og=df

all_dfs = [df_og,df_new]
df=pd.concat(all_dfs).reset_index(drop=True)


    
df.to_csv('vector_final.csv', mode='a', header=False)#df_og is already saved here

delete_row = df[df.iloc[:,1]==""].index
df = df.drop(delete_row)
df_final=df
df_new['Tweet']=[" ".join(review) for review in df_new['Tweet'].values]

df= pd.read_csv('dataset_preprocessed_final.csv', encoding = 'unicode_escape')
df=df.drop('Unnamed: 0',1)

delete_row = df[df.iloc[:,1]==""].index
nan_rows = df[df['Tweet'].isnull()].index
dff = df.drop(nan_rows)

df=dff
pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer='word')),  # strings to token integer counts
    ('tfidf', TfidfTransformer())  # integer counts to weighted TF-IDF scores
      # train on TF-IDF vectors w/ Naive Bayes classifier
])
msg_train, msg_test, label_train, label_test = train_test_split(df['Tweet'], df['Sentiment'], test_size=0.3)
msg_train=pipeline.fit_transform(msg_train)
msg_test=pipeline.transform(msg_test)


nb= MultinomialNB()
nb.fit(msg_train,label_train)
predictions = nb.predict(msg_test)
print(classification_report(predictions,label_test))
print(confusion_matrix(predictions,label_test))
print(accuracy_score(predictions,label_test))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

lr= LogisticRegression()
lr.fit(msg_train,label_train)
predictions = lr.predict(msg_test)
print(classification_report(predictions,label_test))
print(confusion_matrix(predictions,label_test))
print(accuracy_score(predictions,label_test))
#    lr.fit(msg_train,label_train)
#    print ("Accuracy for C=%s: %s" 
#           % (c, accuracy_score(label_test, lr.predict(label_train))))
#    

    




from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC

 
svm= LinearSVC()
svm.fit(msg_train,label_train)
predictions = svm.predict(msg_test)
print(classification_report(predictions,label_test))
print(confusion_matrix(predictions,label_test))
print(accuracy_score(predictions,label_test)) 
#    svm = LinearSVC(C=c)
#    svm.fit(msg_train,label_train)
#    print ("Accuracy for C=%s: %s" 
#           % (c, accuracy_score(label_test, lr.predict(label_train))))
#    
#    
import pickle
pickle.dump(pipeline,open('vector.pkl','wb'))

import pickle
pickle.dump(svm,open('decision.pkl','wb'))

import pickle
pickle.dump(nb,open('decision_nb.pkl','wb'))

import pickle
pickle.dump(lr,open('decision_lr.pkl','wb'))

import pickle
pickle.dump(svm,open('decision_svm.pkl','wb'))