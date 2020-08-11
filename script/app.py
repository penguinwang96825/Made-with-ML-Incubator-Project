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
import time
import random
import os
import spacy
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
traders = get_traders_names()


def get_all_trader_twitter():
    traders = get_traders_names()
    api = TwitterClient().get_twitter_client_api()
    tweet_analyzer = TweetAnalyzer()
    df = pd.DataFrame()
    for trader in tqdm(traders):
        try: 
            tweets = api.user_timeline(screen_name=trader, count=100)
            df_trader = tweet_analyzer.tweets_to_dataframe(tweets)
            df = pd.concat([df, df_trader], axis=0)
        except:
            pass
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
    stop_words = set(stopwords.words('english'))
	font_path = "./Scribble Note DEMO.otf"
	extra_stopwords = ["The", "It", "it", "in", "In", "wh", "yo"]
    stop_words.update(extra_stopwords)
	wordcloud_words = " ".join(clean_tweet)
	wordcloud = WordCloud(
		stopwords=stop_words, height=300, width=500, 
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
	nlp = spacy.load("en_core_web_lg")
	st.markdown("## Top {} retweeted tweets".format(retweeted_k))

	top_5_retweeted_tweets, re_counts = select_top_k_retweeted_tweets(df, k=retweeted_k, print_result=False)
	for i, re_tweet in enumerate(top_5_retweeted_tweets):
		st.markdown("### Top {} (retweeted counts {}):".format(i+1, re_counts[i]))
		# st.markdown(tweet)
		show_sim(re_tweet, key="{}".format(random.randrange(0, 50, 1)), nlp=nlp)
	st.markdown("## Top {} liked tweets".format(liked_k))

	top_5_liked_tweets, li_counts = select_top_k_liked_tweets(df, k=liked_k, print_result=False)
	for i, li_tweet in enumerate(top_5_liked_tweets):
		st.markdown("### Top {} (favorite counts {}):".format(i+1, li_counts[i]))
		# st.markdown(tweet)
		show_sim(li_tweet, key="{}".format(random.randrange(100, 150, 1)), nlp=nlp)


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
		twitter_stream(df, retweeted_k=retweeted_k, liked_k=liked_k)




if __name__ == "__main__":
    caching.clear_cache()
    st.empty()
    main()