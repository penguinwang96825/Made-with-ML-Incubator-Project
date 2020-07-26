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
import string
import nltk
from functools import partial
from collections import Counter
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#consumer_key = os.getenv("consumer_key")
#consumer_secret = os.getenv("consumer_secret")
#access_token = os.getenv("access_token")
#access_token_secret = os.getenv("access_token_secret")

consumer_key=""
consumer_secret=""
access_token=""
access_token_secret=""

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

stopwords = nltk.corpus.stopwords.words('english')
extra_stopwords = {"The", "It", "it", "in", "In", "wh", "yo","atUser","rt","url"}
stopwords.extend(extra_stopwords)
stopwords_final = set(stopwords)

def remove_unicode(text):
    """ Removes unicode strings like "\u002c" and "x96" """
    text = re.sub(r'(\\u[0-9A-Fa-f]+)',r'', text)
    text = re.sub(r'[^\x00-\x7f]',r'',text)
    return text

def replace_URL(text):
    """ Replaces url address with "url" """
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','url',text)
    text = re.sub(r'#([^\s]+)', r'\1', text)
    return text

def replace_at_user(text):
    """ Replaces "@user" with "atUser" """
    text = re.sub('@[^\s]+','atUser',text)
    return text

def remove_hashtag_in_front_of_word(text):
    """ Removes hastag in front of a word """
    text = re.sub(r'#([^\s]+)', r'\1', text)
    return text

# Function for expanding contractions
contractions_dict = {"ain't": "are not","'s":" is","aren't": "are not",
                     "can't": "cannot","can't've": "cannot have",
                     "'cause": "because","could've": "could have","couldn't": "could not",
                     "couldn't've": "could not have", "didn't": "did not","doesn't": "does not",
                     "don't": "do not","hadn't": "had not","hadn't've": "had not have",
                     "hasn't": "has not","haven't": "have not","he'd": "he would",
                     "he'd've": "he would have","he'll": "he will", "he'll've": "he will have",
                     "how'd": "how did","how'd'y": "how do you","how'll": "how will",
                     "I'd": "I would", "I'd've": "I would have","I'll": "I will",
                     "I'll've": "I will have","I'm": "I am","I've": "I have", "isn't": "is not",
                     "it'd": "it would","it'd've": "it would have","it'll": "it will",
                     "it'll've": "it will have", "let's": "let us","ma'am": "madam",
                     "mayn't": "may not","might've": "might have","mightn't": "might not",
                     "mightn't've": "might not have","must've": "must have","mustn't": "must not",
                     "mustn't've": "must not have", "needn't": "need not",
                     "needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
                     "oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
                     "shan't've": "shall not have","she'd": "she would","she'd've": "she would have",
                     "she'll": "she will", "she'll've": "she will have","should've": "should have",
                     "shouldn't": "should not", "shouldn't've": "should not have","so've": "so have",
                     "that'd": "that would","that'd've": "that would have", "there'd": "there would",
                     "there'd've": "there would have", "they'd": "they would",
                     "they'd've": "they would have","they'll": "they will",
                     "they'll've": "they will have", "they're": "they are","they've": "they have",
                     "to've": "to have","wasn't": "was not","we'd": "we would",
                     "we'd've": "we would have","we'll": "we will","we'll've": "we will have",
                     "we're": "we are","we've": "we have", "weren't": "were not","what'll": "what will",
                     "what'll've": "what will have","what're": "what are", "what've": "what have",
                     "when've": "when have","where'd": "where did", "where've": "where have",
                     "who'll": "who will","who'll've": "who will have","who've": "who have",
                     "why've": "why have","will've": "will have","won't": "will not",
                     "won't've": "will not have", "would've": "would have","wouldn't": "would not",
                     "wouldn't've": "would not have","y'all": "you all", "y'all'd": "you all would",
                     "y'all'd've": "you all would have","y'all're": "you all are",
                     "y'all've": "you all have", "you'd": "you would","you'd've": "you would have",
                     "you'll": "you will","you'll've": "you will have", "you're": "you are",
                     "you've": "you have"}

def expand_contractions(text, contractions_dict=contractions_dict):
    # Regular expression for finding contractions
    contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, text)

def remove_digits(text):
    answer = []
    for char in text:
        if not char.isdigit():
            answer.append(char)
    return ''.join(answer)

def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text
def count_slang(text):
    """Creates a dictionary with slangs and their equivalents and replaces them.
    Input: a text
    Output: how many slang words and a list of found slangs.
    """
    with open('slang.txt', encoding="utf-8") as file:
        slang_map = dict(map(str.strip, line.partition('\t')[::2])
        for line in file if line.strip())
    # Longest first for regex
    slang_words = sorted(slang_map, key=len, reverse=True)
    regex = re.compile(r"\b({})\b".format("|".join(map(re.escape, slang_words))))
    replaceSlang = partial(regex.sub, lambda m: slang_map[m.group(1)])
    slangCounter = 0
    slangsFound = []
    tokens = nltk.word_tokenize(text)
    for word in tokens:
        if word in slang_words:
            slangsFound.append(word)
            slangCounter += 1
    return slangCounter, slangsFound
def prep_data(text):
    text = text.lower()
    text = remove_unicode(text)
    text = replace_URL(text)
    text = replace_at_user(text)
    text = remove_hashtag_in_front_of_word(text)
    text = expand_contractions(text)
    text = remove_digits(text)
    text = remove_punct(text)
    return text
    # Word Cloud
def wordcloud(clean_tweet):

    wordcloud_words = " ".join(clean_tweet)
    wordcloud = WordCloud(stopwords = stopwords_final,
    height=300, width=500, background_color="white",
     random_state=100,collocations = False).generate(wordcloud_words)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig("wordcloud.jpg")
    img = Image.open("wordcloud.jpg")
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

    colors = ["teal", "turquoise"]

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

    st.markdown(""" ## Made With ML Incubator """)
    st.markdown(
        """# Twitter Sentiment Analysis"""
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
        tweet_df["clean_tweet"] = tweet_df["tweet"].apply(prep_data)
        tweet_df["slang_count"] = tweet_df["tweet"].apply(count_slang)

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
