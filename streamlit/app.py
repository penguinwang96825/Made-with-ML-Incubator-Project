import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import joblib
from wordcloud import WordCloud
from wordcloud import STOPWORDS

st.title("Analysis of Tweets about Finance, Politics and Forex")
st.sidebar.title("Analysis of Tweets ")

st.markdown("This application is a Streamlit dashboard to analyze tweets")
st.header("Random tweet from selected username")

DATA_URL = ("df_final.gzip")

@st.cache(persist=True)
def load_data():
    data = joblib.load(DATA_URL)
    data = data.iloc[:, 1:]
    return data

data = load_data()

st.sidebar.subheader("Show random tweet")
random_tweet = st.sidebar.radio('username', ('federalreserve', 'economics', 'ecb', 'ftfinancenews'))
st.markdown(data.query("username == @random_tweet")[["tweet"]].sample(n=1).iat[0, 0])

st.sidebar.header("Word Cloud")
word_username = st.sidebar.radio('Display word cloud for what username?', ('federalreserve', 'economics', 'ecb', 'ftfinancenews'))
if not st.sidebar.checkbox("hide", False, key='3'):
    st.subheader('Word cloud for "%s" username' % (word_username))
    df = data[data['username']==word_username]
    words = ' '.join(df['tweet'])
    processed_words = ' '.join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT'])
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', width=800, height=640).generate(processed_words)
    plt.imshow(wordcloud)
    plt.xticks([])
    plt.yticks([])
    st.pyplot()
