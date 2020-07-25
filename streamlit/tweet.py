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
from wordcloud import WordCloud
from textblob import TextBlob
import re
import time
import os

#consumer_key = os.getenv("consumer_key")
#consumer_secret = os.getenv("consumer_secret")
#access_token = os.getenv("access_token")
#access_token_secret = os.getenv("access_token_secret")

consumer_key="yw14UxU2keHgWZmPcpudDMSOz"
consumer_secret="zUFlHBMgCy4NZKh6nSDtly6U0FbqVQOMFIvlJyKPLwksD5jCXQ"
access_token="207458888-8hMxQDGfoceMJLGUusM0Ya9mimdGuNpIHYvmCwko"
access_token_secret="9L32xPjQEMtZckfolkGsDHC7BAPhcJ7xGkUyevUfHDh7P"

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


    return null

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

# subjectivity

def getSubjectivity(tweet):
    sentiment_subjectivity = TextBlob(tweet).sentiment.subjectivity
    return sentiment_subjectivity

# Sentiment score, objectivity and subjectivity

def getSubAnalysis(subjectivity_score):
    if subjectivity_score <= 0.5:
        return "Objective"
    else:
        return "Subjective"

# plot sentiment

def plot_sentiments(tweet_df):
    sentiment_df = (
        pd.DataFrame(tweet_df["sentiment"].value_counts())
        .reset_index()
        .rename(columns={"index": "sentiment_name"})
    )
    fig = go.Figure(
        [go.Bar(x=sentiment_df["sentiment_name"], y=sentiment_df["sentiment"])]
    )
    fig.update_layout(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, title="Sentiment Score"),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig

# plot subjectivity

def plot_subjectivity(tweet_df):
    
    colors = ["mediumturquoise", "blue"]

    fig = go.Figure(
        data=[
            go.Pie(
                values=tweet_df["subjectivity"].values,
                labels=tweet_df["sub_obj"].values,
            )
        ]
    )
    fig.update_traces(
        hoverinfo="label",
        textinfo="percent",
        textfont_size=18,
        marker=dict(colors=colors, line=dict(color="#000000", width=2)),
    )
    return fig

# Streamlit app

def app():
    
    tweet_count = st.empty()
    user_name = st.empty()

    st.sidebar.header("Enter the Details Here!!")

    user_name = st.sidebar.text_area("Enter the Twitter Handle without @")

    tweet_count = st.sidebar.slider(
        "Select the number of Latest Tweets to Analyze", 0, 50, 1
    )

    st.sidebar.markdown(
        "#### Press Ctrl+Enter or Use the Slider to initiate the analysis."
    )
    st.sidebar.markdown(
        "*****************************************************************"
    )

    st.markdown("Made With ML Incubator")
    st.markdown(
        """# Twitter Sentiment Analyzer :slightly_smiling_face: :neutral_face: :angry: """
    )
    st.write(
        "This app analyzes the Twitter tweets and returns the most commonly used words, associated sentiments and the subjectivity score!! Note that Private account / Protected Tweets will not be accessible through this app."
    )
    st.write(
        ":bird: All results are based on the number of Latest Tweets selected on the Sidebar. :point_left:"
    )

    # main
    if user_name != "" and tweet_count > 0:

        with st.spinner("Please Wait!! Analysis is in Progress..."):
            time.sleep(1)

        tweets_list, img_url, name, screen_name, desc = get_tweets(
            user_name, tweet_count
        )

        # adding the retrieved tweet data into a dataframe
        tweet_df = pd.DataFrame([tweet for tweet in tweets_list])
        st.sidebar.success("Twitter Handle Details:")
        st.sidebar.markdown("Name: " + name)
        st.sidebar.markdown("Screen Name: @" + screen_name)
        st.sidebar.markdown("Description: " + desc)

        # calling the function to prep the data
        tweet_df["clean_tweet"] = tweet_df["tweet"]

        # calling the function to create sentiment scoring
        tweet_df["polarity"] = tweet_df["clean_tweet"].apply(getPolarity)
        tweet_df["sentiment"] = tweet_df["polarity"].apply(getAnalysis)
        tweet_df["subjectivity"] = tweet_df["clean_tweet"].apply(getSubjectivity)
        tweet_df["sub_obj"] = tweet_df["subjectivity"].apply(getSubAnalysis)

        # calling the function for plotting the sentiments
        senti_fig = plot_sentiments(tweet_df)
        st.success(
            "Sentiment Analysis for Twitter Handle @"
            + user_name
            + " based on the last "
            + str(tweet_count)
            + " tweet(s)!!"
        )
        st.plotly_chart(senti_fig, use_container_width=True)

        # calling the function for plotting the subjectivity
        subjectivity_fig = plot_subjectivity(tweet_df)

        if sum(tweet_df["subjectivity"].values) > 0:
            st.success(
                "Tweet Subjectivity vs. Objectivity for Twitter Handle @"
                + user_name
                + " based on the last "
                + str(tweet_count)
                + " tweet(s)!!"
            )
            st.plotly_chart(subjectivity_fig, use_container_width=True)
        else:
            st.error(
                "Sorry, too few words to analyze for Subjectivity & Objectivity Score. Please increase the tweet count using the slider on the sidebar for better results."
            )

        # calling the function to create the word cloud
        img = wordcloud(tweet_df["clean_tweet"])
        st.success(
            "Word Cloud for Twitter Handle @"
            + user_name
            + " based on the last "
            + str(tweet_count)
            + " tweet(s)!!"
        )
        st.image(img)

        # displaying the latest tweets
        st.subheader(
            "Latest Tweets (Max 10 returned if more than 10 selected using the sidebar)!"
        )
        st.markdown("*****************************************************************")
        st.success("Latest Tweets from the Twitter Handle @" + user_name)

        length = 10 if len(tweet_df) > 10 else len(tweet_df)

        for i in range(length):
            st.write(
                "Tweet Number: "
                + str(i + 1)
                + ", Tweet Date: "
                + str(tweet_df["date_created"][i])
            )
            st.info(tweet_df["tweet"][i])
    else:
        st.info(
            ":point_left: Enter the Twitter Handle & Number of Tweets to Analyze on the SideBar :point_left:"
        )

# main
if __name__ == "__main__":
    
    caching.clear_cache()
    st.empty()
    app()