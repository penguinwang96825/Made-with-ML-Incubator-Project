import re
import json
import time
import config
import pymongo
import multiprocessing
import tweepy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
from textblob import TextBlob


# ================= Helper Function =================

class TwitterAuth():
    """
    Get authentication for Twitter.
    Get KEY and ACCESS TOKEN from https://developer.twitter.com/en/apps
    from config import CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET
    """
    def auth_twitter_app(self):
        auth = tweepy.OAuthHandler(config.CONSUMER_KEY, config.CONSUMER_SECRET)
        auth.set_access_token(config.ACCESS_TOKEN, config.ACCESS_TOKEN_SECRET)
        return auth

class MaxListener(tweepy.StreamListener):
    """
    This simple stream listener prints status text.
    I save returned tweets into a json file.
    If want to change the format or do other data preprocessing, just revise on process_data().
    """
    def __init__(self, time_limit=60):
        self.start_time = time.time()
        self.limit = time_limit
        
    def on_connect(self):
        print("You are now connected to the streaming API.")
    
    def on_data(self, data):
        if (time.time() - self.start_time) < self.limit:
            self.process_data(data)
            return True
        else:
            return False
    
    def process_data(self, data):
        try:
            # Store raw_data into mongodb
            client = pymongo.MongoClient(MONGO_CLIENT)
            db = client.twitterdb
            datajson = json.loads(data)
            # Insert the data into the mongodb into a collection called twitter_search
            # If twitter_search doesn't exist, it will be created.
            db.twitter_search.insert_one(datajson)
            return True
        
        except BaseException as e:
            print("Error on data: {}".format(e))
            
        return True 
        
    def on_error(self, status_code):
        if status_code == 420:
            return False
        
class TwitterClient():
    """
    Get my own tweets or others.
    
    Parameters:
        twitter_user: if twitter_user is set to None, it means capture my tweets. Instead, crawl twitter_user tweets. 
    """
    def __init__(self, twitter_user=None):
        self.auth = TwitterAuth().auth_twitter_app()
        self.twitter_client = tweepy.API(self.auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
        self.twitter_user = twitter_user
        
    def get_twitter_client_api(self):
        return self.twitter_client
        
    def get_user_timeline_tweets(self, num_tweets):
        tweets = []
        for tweet in tweepy.Cursor(self.twitter_client.user_timeline, id=self.twitter_user).items(num_tweets):
            tweets.append(tweet)
        return tweets
    
    def get_friend_list(self, num_friends):
        friend_list = []
        for friend in tweepy.Cursor(self.twitter_client.friends, id=self.twitter_user).items(num_friends):
            friend_list.append(friend)
        return friend_list
    
    def get_home_timeline_tweets(self, num_tweets):
        home_timeline_tweets = []
        for tweet in tweepy.Cursor(self.twitter_client.home_timeline, id=self.twitter_user).items(num_tweets):
            home_timeline_tweets.append(tweet)
        return home_timeline_tweets

# ================= Main Function =================

class MaxStreamer():
    """
    In Tweepy, an instance of tweepy.Stream establishes a streaming session and routes messages to StreamListener instance.
    This MaxStreamer() class could parse tweets in a time interval.
    Next, it would save the tweets into a json file.
    
    Parameters:
        tweets_filename: json file name.
        time_limit: In default, MaxStreamer will close in 60 sec.
    Return:
        MaxStreamer() class will return a json format file in the current working directory.
    Usage:
        maxstreamer = MaxStreamer(time_limit=60)
        maxstreamer.start(keyword_list=["sheffield"])
    """
    def __init__(self, time_limit=60):
        self.limit = time_limit
        self.auth = TwitterAuth().auth_twitter_app()
        self.listener = MaxListener(time_limit=self.limit)
        self.stream = tweepy.Stream(auth=self.auth, listener=self.listener)
        
    def start(self, keyword_list):
        self.stream.filter(track=keyword_list)
        
class TwitterSearch():
    """
    Twitter Search API.
    This TwitterSearch() class could parse query in a constrained amount.
    
    Usage:
        twitter_searcher = TwitterSearch()
        searched_tweets = twitter_searcher.get_query(query="sheffield", max_tweets=100)
    """
    def __init__(self):
        self.api = TwitterClient().get_twitter_client_api()
    
    def get_query(self, query, max_tweets=100):
        searched_tweets = []
        self.last_id = -1
        self.max_tweets = max_tweets
        self.query = query
        while len(searched_tweets) < self.max_tweets:
            count = self.max_tweets - len(searched_tweets)
            try:
                new_tweets = self.api.search(q=self.query, count=count, max_id=str(self.last_id-1), 
                                        result_type="recent", tweet_mode="extended")
                if not new_tweets:
                    break
                searched_tweets.extend(new_tweets)
                self.last_id = new_tweets[-1].id
                
            except tweepy.TweepError as e:
                print(e)
                break
        client = pymongo.MongoClient(MONGO_CLIENT)
        db = client.twitterdb
        for i in tqdm(range(max_tweets)):
            # Insert the data into the mongodb into a collection called twitter_search
            # If twitter_search doesn't exist, it will be created.
            db.twitter_search.insert_one(searched_tweets[i]._json)
        return searched_tweets
    
class TweetAnalyzer():
    """
    An analyzer to tweets.
    
    Usage: 
        api = TwitterClient().get_twitter_client_api()
        tweets = api.user_timeline(screen_name="sheffield", count=100)
        df = tweet_analyzer.tweets_to_dataframe(tweets)
    """
    def clean_tweet(self, text):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())
    
    def get_polarity(self, text):
        try:
            textblob = TextBlob(unicode(self.clean_tweet(text), 'utf-8'))
            polarity = textblob.sentiment.polarity
        except:
            polarity = 0.0
        return polarity
    
    def get_subjectivity(self, text):
        try:
            textblob = TextBlob(unicode(self.clean_tweet(text), 'utf-8'))
            subjectivity = textblob.sentiment.subjectivity
        except:
            subjectivity = 0.0
        return subjectivity
    
    def tweets_to_dataframe(self, tweets):
        df = pd.DataFrame()
        df["tweet"] = np.array([tweet.text for tweet in tweets])
        df["polarity"] = df['tweet'].apply(self.get_polarity)
        df["subjectivity"] = df['tweet'].apply(self.get_subjectivity)
        df['word_count'] = np.array([len(tweet.text) for tweet in tweets])
        df['char_count'] = df['tweet'].apply(lambda x : len(x.replace(" ","")))
        df['word_density'] = df['word_count'] / (df['char_count'] + 1)
        df["id"] = np.array([tweet.id for tweet in tweets])
        df["favorite_count"] = np.array([tweet.favorite_count for tweet in tweets])
        df["retweet_count"] = np.array([tweet.retweet_count for tweet in tweets])
        df["date"] = np.array([tweet.created_at for tweet in tweets])
        df["source"] = np.array([tweet.source for tweet in tweets])
        return df