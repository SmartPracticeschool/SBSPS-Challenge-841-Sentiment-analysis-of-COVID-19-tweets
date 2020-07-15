from twython import Twython,TwythonRateLimitError,TwythonError
import pandas as pd
import itertools 
import time

CONSUMER_KEY = "8Y7JmfyzAGoN4ynp38OBokbad"
CONSUMER_SECRET = "w4rWBwRH1YbbRYyVu3oRLLDfb0GtEYYYoIniAoqxFm4jkijhoQ"
OAUTH_TOKEN = "1272828877954404354-KziDz09I8t9LFIoq5LFUPSqfm4iUzP"
OAUTH_TOKEN_SECRET = "83jssKdgUYhzyBRZbof4Eys0FTWXr0rAO1xaTNucAz5S6"
twitter = Twython(
    CONSUMER_KEY, CONSUMER_SECRET,
    OAUTH_TOKEN, OAUTH_TOKEN_SECRET)

ds=pd.read_csv('/Users/sanjanasrinivasareddy/Documents/ibm-hack/corona_tweets_70.csv')
ds=ds.iloc[5000:5900,:]
#x=ds['Id']
x=[]
a=0
for i,j in ds.iterrows():
    #print(j['Id'])
    print(a)
    a=a+1
    try:
        tweet = twitter.show_status(id=int(j['Id']))
        print(int(j['Id']))
        print(tweet['text'])
        #ds['Tweet']=tweet['text']
        x.append(tweet['text'])
#    tweet = twitter.show_status(id=int(j['Id']))
#    print(int(j['Id']))
#    print(tweet['text'])
#    #ds['Tweet']=tweet['text']
#    x.append(tweet['text'])
    except TwythonRateLimitError as error:
        remainder = float(twitter.get_lastfunction_header(header='x-rate-limit-reset')) - time.time()
        time.sleep(remainder)
        tweet = twitter.show_status(id=int(j['Id']))
        print(int(j['Id']))
        print(tweet['text'])
        #ds['Tweet']=tweet['text']
        x.append(tweet['text'])
        continue
    except TwythonError as error:
        print("")
        #ds['Tweet']=""
        x.append("")
        
    except:
        print("")
        #ds['Tweet']=""
        x.append("")
#x.append("")

ds['Tweet']=x
#    try:
#        tweet = twitter.show_status(id=j['Id'])
#        #print(tweet['text'])
#        ds['Tweet']=tweet['text']
#    except:
ds.to_csv('file6.csv', mode='a', header=False)
