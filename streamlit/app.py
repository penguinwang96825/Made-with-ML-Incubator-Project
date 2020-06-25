import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
import joblib

st.title("Sentiment Analysis of Tweets about the Forex")
st.sidebar.title("Sentiment Analysis of Tweets about Forex")

st.markdown("This application is a Streamlit dashboard to analyze Forex")

DATA_URL = ("df_final.gzip")

@st.cache(persist=True)
def load_data():
    data = joblib.load(DATA_URL)
    data = data.iloc[:, 1:]
    return data

data = load_data()

