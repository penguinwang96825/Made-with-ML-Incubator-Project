import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import joblib
from wordcloud import WordCloud
from wordcloud import STOPWORDS
import requests
from datetime import datetime

api_token = 'ucscrG7YXCFz1Xp52lPk4GOqr5TyzWbRmxqrPYamcyCRke9l1RPu0pXRJa7e'

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

@st.cache(persist=True)
def load_url():
    url = 'https://api.worldtradingdata.com/api/v1/forex_history'
    params = {
    'base': 'USD', 
    'convert_to': 'GBP', 
    "date_from": "2020-06-06", 
    'date_to': "2020-06-14", 
    'api_token': api_token}
    response = requests.request('GET', url, params=params)
    return response.text



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

forex_data = load_url()
st.write(forex_data)



currency = st.sidebar.radio('currency',('USD'))
ticker = forex_data[forex_data['Name'] == currency, 'Ticker']
end_date = st.sidebar.date_input('end date', value=datetime.now()).strftime("%Y-%m-%d")
start_date = st.sidebar.date_input('start date', value=datetime(2010, 5, 31)).strftime("%Y-%m-%d")