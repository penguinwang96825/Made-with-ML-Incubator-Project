import streamlit as st
from streamlit import caching
from PIL import Image
import PIL
import io
import requests
import sys
import tweepy
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from wordcloud import WordCloud
from textblob import TextBlob
import re
import time
import os

consumer_key = os.getenv("consumer_key")
consumer_secret = os.getenv("consumer_secret")
access_token = os.getenv("access_token")
access_token_secret = os.getenv("access_token_secret")

# creating the authentication object, setting access token and creating the api object
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)


# get tweets
def get_tweets(user_name, tweet_count):
    
    tweets_list = []
    img_url = ""
    name = ""

    try:
        for tweet in api.user_timeline(
            id=user_name, count=tweet_count, tweet_mode="extended"
        ):
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
            "Failed to retrieve the Tweets. Please check if the twitter handle is correct. "
        )
        sys.exit(1)

    return tweets_list, img_url, name, screen_name, desc

# preprocessing


extra_stopwords = ["The", "It", "it", "in", "In", "wh"]


def prep_data(tweet):

    # cleaning the data
    tweet = re.sub("https?:\/\/\S+", "", tweet)  # replacing url with domain name
    tweet = re.sub("#[A-Za-z0–9]+", " ", tweet)  # removing #mentions
    tweet = re.sub("#", " ", tweet)  # removing hash tag
    tweet = re.sub("\n", " ", tweet)  # removing \n
    tweet = re.sub("@[A-Za-z0–9]+", "", tweet)  # removing @mentions
    tweet = re.sub("RT", "", tweet)  # removing RT
    tweet = re.sub("^[a-zA-Z]{1,2}$", "", tweet)  # removing 1-2 char long words
    tweet = re.sub("\w*\d\w*", "", tweet)  # removing words containing digits
    for word in extra_stopwords:
        tweet = tweet.replace(word, "")

    # lemmitizing
    lemmatizer = WordNetLemmatizer()
    new_s = ""
    for word in tweet.split(" "):
        lemmatizer.lemmatize(word)
        if word not in stopwords.words("english"):
            new_s += word + " "

    return new_s[:-1]

    # Word Cloud
def wordcloud(clean_tweet):
    
    wordcloud_words = " ".join(clean_tweet)
    wordcloud = WordCloud(
        height=300, width=500, background_color="black", random_state=100,
    ).generate(wordcloud_words)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig("cloud.jpg")
    img = Image.open("cloud.jpg")
    return img

# polarity

def getPolarity(tweet):
    sentiment_polarity = TextBlob(tweet).sentiment.polarity
    return sentiment_polarity


def getAnalysis(polarity_score):
    if polarity_score < 0:
        return "Negative"
    elif polarity_score == 0:
        return "Neutral"
    else:
        return "Positive"