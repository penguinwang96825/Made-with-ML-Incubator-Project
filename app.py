import PIL
import io
import sqlite3
import requests
import sys
import config
import tweepy
import nltk
import string
import re
import json
import time
import random
import os
import pathlib
import spacy
import tweepy
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import spacy_streamlit

from spacy import displacy
from bs4 import BeautifulSoup
from streamlit import caching
from os import path
from PIL import Image
from plotly.subplots import make_subplots
from nltk import *
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from wordcloud import WordCloud
from wordcloud import STOPWORDS
from wordcloud import ImageColorGenerator
from textblob import TextBlob


consumer_key = config.CONSUMER_KEY
consumer_secret = config.CONSUMER_SECRET
access_token = config.ACCESS_TOKEN
access_token_secret = config.ACCESS_TOKEN_SECRET


# Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)


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


@st.cache(persist=True)
def get_traders_names():
    headers = {
        "User-Agent": 
        "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.105 Mobile Safari/537.36"}
    url = "https://www.forexcrunch.com/60-top-forex-twitter-accounts/"
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")
    table = soup.find(name="ol")
    traders = []
    for element in table.find_all(name="li"):
        trader = element.find(name="a").text.replace("@", "")
        traders.append(trader)
    return traders


@st.cache(persist=True)
def get_all_trader_twitter():
	traders = []
	for line in open('trader.txt','r').readlines():
		line = line.strip()
		traders.append(line)
	api = TwitterClient().get_twitter_client_api()
	tweet_analyzer = TweetAnalyzer()
	df = pd.DataFrame()
	for i, trader in enumerate(traders):
		try: 
			tweets = api.user_timeline(screen_name=trader, count=100)
			df_trader = tweet_analyzer.tweets_to_dataframe(tweets)
			df = pd.concat([df, df_trader], axis=0)
		except:
			print(f"Can't get {trader} tweets timeline.")
	return df


def insert_into_db():
    df = get_all_trader_twitter()
    print(df.shape)
    conn = sqlite3.connect('twitter.db')
    cursor = conn.cursor()
    cursor.execute("""DROP TABLE IF EXISTS TEMPTRADER""")
    df.to_sql(name='TEMPTRADER', con=conn, if_exists="replace", index=False)
    cursor.execute(
        '''
        INSERT INTO TRADER
        SELECT A.* FROM TEMPTRADER A
        LEFT JOIN TRADER B 
        ON A.id=B.id 
        WHERE B.id IS NULL
        ''')
    conn.commit()
    conn.close()
 

@st.cache(persist=True)
def read_from_db():
    conn = sqlite3.connect('./twitter.db')
    df = pd.read_sql_query("SELECT * FROM TRADER", con=conn)
    return df


def get_tweets(user_name, tweet_count):
    tweets_list = []
    img_url = ""
    name = ""
    try:
        for tweet in api.user_timeline(
            id=user_name, count=tweet_count, tweet_mode="extended"):
            tweets_dict = {}
            tweets_dict["date_created"] = tweet.created_at
            tweets_dict["tweet_id"] = tweet.id
            tweets_dict["tweet"] = tweet.full_text
            tweets_list.append(tweets_dict)

        img_url = tweet.user.profile_image_url
        name = tweet.user.name
        screen_name = tweet.user.screen_name
        desc = tweet.user.description

    except BaseException as e:
        st.exception(
            "Failed to retrieve the Tweets. Please check if the twitter handle is correct.")
        sys.exit(1)

    return tweets_list, img_url, name, screen_name, desc


def prep_data(tweet):
    return null


def wordcloud(clean_tweet):
    font_path = "./Scribble Note DEMO.otf"
    extra_stopwords = ["The", "It", "it", "in", "In", "wh", "yo", "RT"]
    for n in extra_stopwords:
        STOPWORDS.add(n)
    wordcloud_words = " ".join(clean_tweet)
    wordcloud = WordCloud(
        stopwords=STOPWORDS, height=300, width=500, 
        background_color="white", random_state=100, font_path=font_path
    ).generate(wordcloud_words)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig("wordcloud.jpg")
    img = Image.open("wordcloud.jpg")
    return img


def get_polarity(tweet):
    return TextBlob(tweet).sentiment.polarity


def get_analysis(polarity_score):
    if polarity_score < 0:
        return "Negative"
    elif polarity_score == 0:
        return "Neutral"
    else:
        return "Positive"


def get_subjectivity(tweet):
    return TextBlob(tweet).sentiment.subjectivity


def get_sub_analysis(subjectivity_score):
    if subjectivity_score <= 0.5:
        return "Objective"
    else:
        return "Subjective"


def plot_sentiments(tweet_df):
    sentiment_df = (
        pd.DataFrame(tweet_df["sentiment"].value_counts())
        .reset_index()
        .rename(columns={"index": "sentiment_name"}))
    fig = go.Figure(
        [go.Bar(x=sentiment_df["sentiment_name"], y=sentiment_df["sentiment"])])
    fig.update_layout(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, title="Sentiment Score"),
        plot_bgcolor="rgba(0,0,0,0)",)
    return fig


def plot_subjectivity(tweet_df):
    colors = ["teal", "turquoise"]
    fig = go.Figure(
        data=[
            go.Pie(
                values=tweet_df["subjectivity"].values,
                labels=tweet_df["sub_obj"].values,)])
    fig.update_traces(
        hoverinfo="label",
        textinfo="percent",
        textfont_size=18,
        marker=dict(colors=colors, line=dict(color="#000000", width=2)),)
    return fig


def polarity_plot(polarity_df): 
    z = np.sort(np.asarray(polarity_df))
    plt.figure(figsize= (15, 10))
    fig = make_subplots(
        rows=2, 
        cols=1,
        subplot_titles=("Heatmap of user sentiments(Polarity)",
                        "Sentiment Distribution(Polarity)"))
   # Heatmap
    fig.add_trace(go.Heatmap(
        z= [z],
        type='heatmap',
        colorscale='Viridis', zmax=1, zmin=-1,
        showscale=False), row=1, col=1)

    # Histogram
    fig.add_trace(go.Histogram(
        x= polarity_df,
        name='polarity',
        xbins=dict(start=-1.0, end=1.0)),row=2, col=1)

    fig.update_layout(
        autosize=False,
        width=800,
        height=500, 
        title_x=0.5, 
        title_text='Sentiment Distribution')
    fig.update_traces(opacity=0.75)
    return fig


def subjectivity_plot(subjectivity_df):
    z = np.sort(np.asarray(subjectivity_df))
    plt.figure(figsize=(15, 10))

    fig = make_subplots(
        rows=2, cols=1, 
        subplot_titles=(
            "Heatmap of user sentiments(Subjectivity)", 
            "Sentiment Distribution(Subjectivity)"))
    # Heatmap  
    fig.add_trace(go.Heatmap(
        z=[z], 
        type='heatmap', name='subjectivity',
        colorscale='Viridis', zmax=1, zmin=-1, 
        showscale=False), row=1, col=1)

    # Histogram
    fig.add_trace(go.Histogram(
        x=subjectivity_df, 
        name='subjectivity(0-1)',
        xbins=dict(start=0, end=1.0)), row=2, col=1)

    fig.update_layout(
        autosize=False,
        width=800,
        height=500, 
        title_x=0.5, 
        title_text='Sentiment Distribution')
    return fig


def add_frequency(ax, data):
    ncount = len(data)
    ax2 = ax.twinx()
    ax2.yaxis.tick_left()
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    ax2.yaxis.set_label_position('left')
    ax2.set_ylabel('Frequency [%]')
    ax2.set_ylim(0, 100)
    ax2.grid(None)

def upper_rugplot(data, height=.05, ax=None, **kwargs):
    from matplotlib.collections import LineCollection
    ax = ax or plt.gca()
    kwargs.setdefault("linewidth", 1)
    segs = np.stack((np.c_[data, data],
                     np.c_[np.ones_like(data), np.ones_like(data)-height]),
                    axis=-1)
    lc = LineCollection(segs, transform=ax.get_xaxis_transform(), **kwargs)
    ax.add_collection(lc)


def plot_favorite_count(df):
	plt.figure(figsize=(16, 5))
	fig = sns.countplot("favorite_count", data=df, palette=sns.color_palette("hls", 8))
	plt.title("Favorite Count")
	plt.xlim((-0.5, 9.5))
	plt.grid(axis="y")
	add_frequency(fig, df)
	return fig


def select_top_k_retweeted_tweets(df, k=5, print_result=False):
    tweet_df = df.sort_values(by='retweet_count', ascending=False)
    tweet_df = tweet_df.reset_index(drop=True)
    top_k_retweeted_tweets = []
    counts = []
    for i in range(k):
        top_k_retweeted_tweets.append(tweet_df['tweet'].iloc[i])
        counts.append(tweet_df['retweet_count'].iloc[i])
    if print_result:
        # Total tweets
        print('Total tweets this period:', len(df.index))
        print('='*30, "\n")

        # Retweets
        print('Mean retweets:', round(tweet_df['retweet_count'].mean(), 2), '\n')
        print('Top 5 retweeted tweets:')
        print('-'*25)
        for i in range(5):
            print("#{}: ".format(i+1), tweet_df['retweet_count'].iloc[i], "\n", tweet_df['tweet'].iloc[i], "\n")
        print('\n')

    return top_k_retweeted_tweets, counts

def select_top_k_liked_tweets(df, k=5, print_result=False):
    tweet_df = df.sort_values(by='favorite_count', ascending=False)
    tweet_df = tweet_df.reset_index(drop=True)
    top_k_liked_tweets = []
    counts = []
    for i in range(k):
        top_k_liked_tweets.append(tweet_df['tweet'].iloc[i])
        counts.append(tweet_df['favorite_count'].iloc[i])
    if print_result:
        # Total tweets
        print('Total tweets this period:', len(df.index))
        print('='*30, "\n")

        # Likes
        print('Mean likes:', round(tweet_df['favorite_count'].mean(), 2), '\n')
        print('Top 5 liked tweets:')
        print('-'*25)
        for i in range(5):
            print("#{}: ".format(i+1), tweet_df['favorite_count'].iloc[i], "\n", tweet_df['tweet'].iloc[i], "\n")
        print('\n')

    return top_k_liked_tweets, counts


def show_sim(text, key, nlp):
	spacy_streamlit.visualize_similarity(nlp, (str(text), "forex"), key=key)


def eda_on_tweet(user_name, tweet_count):
    if user_name != "" and tweet_count > 0:

        with st.spinner("Please Wait!! Analysis is in Progress..."):
            time.sleep(1)

        tweets_list, img_url, name, screen_name, desc = get_tweets(
            user_name, tweet_count)

        # Adding the retrieved tweet data into a dataframe
        tweet_df = pd.DataFrame([tweet for tweet in tweets_list])
        st.sidebar.success("Twitter Handle Details: ")
        st.sidebar.markdown("Name: " + name)
        st.sidebar.markdown("Screen Name: @" + screen_name)
        st.sidebar.markdown("Description: " + desc)

        # Calling the function to prep the data
        tweet_df["clean_tweet"] = tweet_df["tweet"]

        # Calling the function to create sentiment scoring
        tweet_df["polarity"] = tweet_df["clean_tweet"].apply(get_polarity)
        tweet_df["sentiment"] = tweet_df["polarity"].apply(get_analysis)
        tweet_df["subjectivity"] = tweet_df["clean_tweet"].apply(get_subjectivity)
        tweet_df["sub_obj"] = tweet_df["subjectivity"].apply(get_sub_analysis)

        # Calling the function for plotting the sentiments
        senti_fig = plot_sentiments(tweet_df)
        st.success(
            "Sentiment Analysis for Twitter Handle @"
            + user_name
            + " based on the last "
            + str(tweet_count)
            + " tweet(s)!!")
        st.plotly_chart(senti_fig, use_container_width=True)

        # Calling the function for plotting the subjectivity
        subjectivity_fig = plot_subjectivity(tweet_df)

        if sum(tweet_df["subjectivity"].values) > 0:
            st.success(
                "Tweet Subjectivity vs. Objectivity for Twitter Handle @"
                + user_name
                + " based on the last "
                + str(tweet_count)
                + " tweet(s)!!")
            st.plotly_chart(subjectivity_fig, use_container_width=True)
        else:
            st.error(
                "Sorry, too few words to analyze for Subjectivity & Objectivity Score. \
                Please increase the tweet count using the slider on the sidebar for better results.")

        # Calling the function to create the word cloud
        img = wordcloud(tweet_df["clean_tweet"])
        st.success(
            "Word Cloud for Twitter Handle @"
            + user_name
            + " based on the last "
            + str(tweet_count)
            + " tweet(s)!!")
        st.image(img)

        # Displaying the latest tweets
        st.subheader(
            "Latest Tweets (Max 10 returned if more than 10 selected using the sidebar)!")
        st.markdown("*****************************************************************")
        st.success("Latest Tweets from the Twitter Handle @" + user_name)

        length = 10 if len(tweet_df) > 10 else len(tweet_df)
        for i in range(length):
            st.write(
                "Tweet Number: "
                + str(i + 1)
                + ", Tweet Date: "
                + str(tweet_df["date_created"][i]))
            st.info(tweet_df["tweet"][i])
    else:
        st.info(
            ":point_left: Enter the Twitter Handle & Number of Tweets to Analyze on the SideBar :point_left:")


def twitter_stream(df, retweeted_k=5, liked_k=5):
	# nlp = spacy.load("en_core_web_lg")
	# # nlp = en_core_web_lg.load()
	# st.markdown("## Top {} retweeted tweets".format(retweeted_k))

	# top_5_retweeted_tweets, re_counts = select_top_k_retweeted_tweets(df, k=retweeted_k, print_result=False)
	# for i, re_tweet in enumerate(top_5_retweeted_tweets):
	# 	st.markdown("### Top {} (retweeted counts {}):".format(i+1, re_counts[i]))
	# 	# st.markdown(tweet)
	# 	show_sim(re_tweet, key="{}".format(random.randrange(0, 50, 1)), nlp=nlp)
	# st.markdown("## Top {} liked tweets".format(liked_k))

	# top_5_liked_tweets, li_counts = select_top_k_liked_tweets(df, k=liked_k, print_result=False)
	# for i, li_tweet in enumerate(top_5_liked_tweets):
	# 	st.markdown("### Top {} (favorite counts {}):".format(i+1, li_counts[i]))
	# 	# st.markdown(tweet)
	# 	show_sim(li_tweet, key="{}".format(random.randrange(100, 150, 1)), nlp=nlp)
	pass


@st.cache
def load_forex():
    url = "https://bloomberg-market-and-financial-news.p.rapidapi.com/market/get-cross-currencies"

    querystring = {"id":"eur%2Cgbp%2Cjpy%2Cusd"}

    headers = {
        'x-rapidapi-host': "bloomberg-market-and-financial-news.p.rapidapi.com",
        'x-rapidapi-key': "537b661c9emsh7d187d9ef48a44bp17c786jsnc8b4ab8c52b4"
        }

    response = requests.request("GET", url, headers=headers, params=querystring)
    res = json.loads(response.text)
    result = res.get("result")
    data = pd.DataFrame(result)
    return data


@st.cache(persist=True)
def currency_snippets(currency, start_date='2018-07-02'):
    """
    Reference from https://fxmarketapi.com/documentation
        "USDAED": "United Arab Emirates Dirham",
        "USDARS": "Argentine Peso",
        "AUDUSD": "Australian Dollar",
        "USDBRL": "Brazilian Real",
        "BTCUSD": "Bitcoin",
        "USDCAD": "Canadian Dollar",
        "USDCHF": "Swiss Franc",
        "USDCLP": "Chilean Peso",
        "USDCNY": "Chinese Yuan",
        "USDCOP": "Colombian Peso",
        "USDCZK": "Czech Republic Koruna",
        "USDDKK": "Danish Krone",
        "EURUSD": "Euro",
        "GBPUSD": "British Pound Sterling"
    """
    URL = "https://fxmarketapi.com/apipandas"
    params = {
        'currency' : '{}'.format(currency),
        'start_date' : '{}'.format(start_date), 
        'api_key':'SQHMG9v7PRfclvV0iLGA'}

    response = requests.get("https://fxmarketapi.com/apipandas", params=params)
    df = pd.read_json(response.text)
    return df


def forex_prediction():
    currency = st.sidebar.selectbox(
        "Which do you like the most?",
        ("AUDUSD","BTCUSD","USDCAD", "USDCHF", "USDCNY", "EURUSD", "GBPUSD"))
    df = currency_snippets(currency=currency, start_date='2018-07-02', end_date='2020-07-02')
    return df


def main():
    df = read_from_db()
    activities = ["Exploratory Data Analysis", "Twitter Stream", "Forex Prediction", "Backtesting"]
    choice = st.sidebar.selectbox("Choose Activity", activities)

    if choice == "Exploratory Data Analysis":
        # Basic info
        st.sidebar.header("Enter the Details Here!!")
        user_name = st.sidebar.text_area(r"Enter the Twitter Handle without @")
        tweet_count = st.sidebar.slider(
            r"Select the number of Latest Tweets to Analyze", 0, 50, 1)

        st.sidebar.markdown(
            "#### Press Ctrl+Enter or Use the Slider to initiate the analysis.")
        st.sidebar.markdown(
            "*****************************************************************")

        st.markdown("""## Made With ML Incubator """)
        st.markdown("""# Twitter Sentiment Analysis""")
        st.write(
            "This app analyzes the Twitter tweets and returns the most commonly used words, \
            associated sentiments and the subjectivity score!! Note that Private account or \
            Protected Tweets will not be accessible through this app.")
        st.write(
            ":bird: All results are based on the number of Latest Tweets selected on the \
            Sidebar. :point_left:")

        eda_on_tweet(user_name, tweet_count)

    if choice == "Twitter Stream":
        retweeted_k = st.sidebar.slider(r"Select top k retweeted tweets", 0, 10, 1)
        liked_k = st.sidebar.slider(r"Select top k liked tweets", 0, 10, 1)

        if st.button('Say hello'):
        	insert_into_db()
        	df = read_from_db()
        	st.balloons()
        	st.dataframe(df)
        	# twitter_stream(df, retweeted_k=5, liked_k=5)

    if choice == "Forex Prediction":
        st.markdown("""## Made With ML Incubator """)
        st.markdown("""# Twitter Forex Prediction""")
        currency = st.sidebar.selectbox(
            "Which do you like the most?",
            ("AUDUSD","BTCUSD","USDCAD", "USDCHF", "USDCNY", "EURUSD", "GBPUSD"))
        df = currency_snippets(currency)
        st.markdown("### Close Price Historical Data Plot")
        st.line_chart(df.close)
        

    if choice == "Backtesting":
        pass


if __name__ == "__main__":
    caching.clear_cache()
    st.empty()
    main()
