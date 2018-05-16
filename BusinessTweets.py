#The link with content pertinent to this exercise is https://dev.to/rodolfoferro/sentiment-analysis-on-trumpss-tweets-using-python-
#First import all the relevant modules to perform sentiment analysis
#If errors, install using pip. Import Tweepy, pandas and numpy
import tweepy
import pandas as pd
import numpy as np
import textblob

#Install IPython before doing this. Then continue to import
from IPython.display import display
#Matplotlib is a Python 2D plotting library which produces publication quality
#figures in a variety of hardcopy formats and interactive environments across platforms.
import matplotlib.pyplot as plt
#Seaborn is a Python visualization library based on matplotlib.
#It provides a high-level interface for drawing attractive statistical graphics.
import seaborn as sns
#instead of the %matplotlib inline, generates plots on the same page
#-----------------------------------------------------
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
#----------------------------------------------------
#We import our access keys:
from credentials import*

#API's setup: This is important

def twitter_setup():
    """
    Utility function to setup the Twitter's API with our access keys provided.

    """
    #Authentication and access using keys:
    #Call OAuthHandler function in tweepy module with arguments Consumer key     #and consumer secret, then assign to variable auth
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    #of what's returned above and assigned to auth, call set_access_token fn
    #with variables access token and access secret.
    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

    #Return API with authentication:
    api = tweepy.API(auth)
    return api

# we create an extractor object:
extractor  = twitter_setup()
#Tweepy function extract from screen_name's user the quantity of count tweets.
#instead of Trump, chose Bloomberg which has the handle business
tweets = extractor.user_timeline(screen_name = "business", count = 150)
print("Number of tweets extracted: {}.\n".format(len(tweets)))

#Then we print the most recent 5 tweets
#tweet is just the counter variable, it mught as well be i or whatever you like
print("5 recent tweets:\n")
for tweet in tweets[:5]:
    print(tweet.text)
    print()

#Each element in that list is a #tweet object from Tweepy, and we will learn how to handle this data in the next #subsection.


#Create Panadas dataframe as follows, using list comprehension

data = pd.DataFrame(data = [tweet.text for tweet in tweets], columns = ['Tweets'])

#Display the first 10 elements of the dataframe:
display(data.head(10))
#dir returns a list of the attributes and methods of any object: modules, functions, strings, lists, dictionaries... pretty much anything
print(dir(tweets[0]))



#A single tweet contains a lot of metadata
#Here we pull out that data

print(tweets[0].id)
print(tweets[0].created_at)
print(tweets[0].source)
print(tweets[0].favorite_count)
print(tweets[0].geo)
print(tweets[0].coordinates)
print(tweets[0].entities)

#Use Python list comprehension, to add relevant data:
data['len'] = np.array([len(tweet.text) for tweet in tweets])
data['ID']  = np.array([tweet.id for tweet in tweets])
data['Date'] = np.array([tweet.created_at for tweet in tweets])
data['Source'] = np.array([tweet.source for tweet in tweets])
data['Likes'] = np.array([tweet.favorite_count for tweet in tweets])
data['RTs'] = np.array([tweet.retweet_count for tweet in tweets])

#Dispay of the changes made above, Display of the first 10 elements from dataframe:
display(data.head(12))

#basic statistical data, such as the mean of the length of characters of all tweets, #the tweet with more likes and retweets, etc.

#We extend the mean of lengths

mean = np.mean(data['len'])
print("The length's avaerage in tweets: {}".format(mean))

#More Pandas functionality
#We extract the tweet with more FAV's and more RTs:
#Uses numpy function max
#This guves the number of likes and retweets
fav_max = np.max(data['Likes'])
rt_max = np.max(data['RTs'])

#This gives the location of those tweets
fav = data[data.Likes == fav_max].index[0]
rt = data[data.RTs == rt_max].index[0]

#Max FAVs:
#This gives the actual tweets with text etc, by scoping out the dataframe
print("The tweet with more likes is: \n{}".format(data['Tweets'][fav]))
print("Number of likes: {}".format(rt_max))
print("{} characters.\n".format(data['len'][fav]))


#Max RTs:
print("The tweet with more retweets is: \n{}".format(data['Tweets'][rt]))
print("Number of retweets: {}".format(rt_max))
print("{} charcters.\n".format(data['len'][rt]))

#Pandas time series object
#Create time series with respect to tewwt lengths, likes and retweets

tlen = pd.Series(data= data['len'].values, index = data['Date'])
tfav = pd.Series(data=data['Likes'].values, index = data['Date'])
tret = pd.Series(data = data['RTs'].values, index = data['Date'])

#Plot the time series using the methods in the time series object in Pandas
#Plot the tweet lengths wrt time
tlen.plot(figsize = (16, 4), color = 'r');
# Likes vs retweets visualization:
tfav.plot(figsize=(16,4), label="Likes", legend=True)
tret.plot(figsize=(16,4), label="Retweets", legend=True);



#Let's make a pie chart
#Clean all the sources
sources = []
for source in data['Source']:
    if source not in sources:
        sources.append(source)

#We print sources list:
print("Creation of content sources:")
for source in sources:
    print("* {}".format(source))

#now we get to visualizing the pie chart
percent = np.zeros(len(sources))

for source in data['Source']:
    for index in range(len(sources)):
        if source == sources[index]:
            percent[index] +=1
            pass
percent /=100

pie_chart = pd.Series(percent, index =sources, name = 'Sources')
pie_chart.plot.pie(fontsize =11, autopct = '%.2f', figsize =(6,6));

#Sentiment analysis
#library re which uses
#Now we get into cleaning text
#cleaning:  splitting it into words and handling punctuation and case
#create a classifier to analyze the polarity of each tweet after cleaning the text in it

from textblob import TextBlob
import re

def clean_tweet(tweet):
    '''
    Utility function to clean the text in a tweet by removing
    links and special characters using regex.
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\s+)","",tweet).split())

#textblob already provides a trained analyzer (cool, right?). Textblob can work with different machine #learning models used in natural language processing.

def analyze_sentiment(tweet):
    '''
    Utility function to classify the polarity of a tweet using textblob
    '''
    analysis = TextBlob(clean_tweet(tweet))
    if analysis.sentiment.polarity == 0:
        return 1

    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1

#but for the purposes of the startup please, work on training your own models

#Adding an extra column to the data
data['SA'] = np.array([analyze_sentiment(tweet) for tweet in data['Tweets']])

#We display the updated data frame with the new column, just the first 10 rows:
display(data.head(10))
#Analysis
#Construct lists with classified tweets:
pos_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index]>0]
neu_tweets = [tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index]==0]
neg_tweets = [tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index]<0]

#We print percentages
print("Percentage of positive tweets: {}%".format(len(pos_tweets)*100/len(data['Tweets'])))
print("Percentage of neutral tweets: {}%".format(len(neu_tweets)*100/len(data['Tweets'])))
print("Percentage of negative tweets: {}%".format(len(neg_tweets)*100/len(data['Tweets'])))
