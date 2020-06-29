import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
import joblib

st.title("Analysis of Tweets about Forex")
st.sidebar.title("Analysis of Tweets about Forex")

st.markdown("This application is a Streamlit dashboard to analyze Forex")

DATA_URL = ("df_final.gzip")

@st.cache(persist=True)
def load_data():
    data = joblib.load(DATA_URL)
    data = data.iloc[:, 1:]
    return data

data = load_data()

st.sidebar.subheader("Show random tweet")
random_tweet = st.sidebar.radio('username', ('pafxss', 'federalreserve', 'economics'))
st.sidebar.markdown(data.query("username == @random_tweet")[["tweet"]].sample(n=1).iat[0, 0])

st.sidebar.button("Show tweets")
st.markdown(data.query(("likes_count>=1000")))

