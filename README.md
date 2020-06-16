# Made_with_ML_Incubator_Project
In this project, we focus on the researches applying natural language processing (NLP) technologies in the finance domain. 

## Introduction
Sentiment analysis can use natural language processing, artificial intelligence, text analysis and computational linguistics to identify the attitude of several topics. In this project, we focus on the researches applying natural language processing (NLP) technologies in the finance domain. First, we will dig into some people who have huge impact on financial market. Second, we will predict foreign exchange rates by making use of the trending topics from Twitter, using a machine learning based model.

## Collect Data

### Import Packages
```python
# An advanced Twitter scraping & OSINT tool written in Python that doesn't use Twitter's API.
import twint

# Solve compatibility issues with notebooks and RunTime errors.
import nest_asyncio
import os
import sys
sys.path.append("twint/")
nest_asyncio.apply()
%load_ext autoreload
%autoreload 2

# Python preprocessing library.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
```

### Target Tweets
Forex trading is fast, very fast, and Twitter fits like a glove to any forex trader’s hand. There’s lots of quick and useful information coming in the form of tweets and sometimes too much information. We got a list of top forex twitter accounts from [here](https://www.forexcrunch.com/60-top-forex-twitter-accounts/), each one coming with different characteristics, to suit traders interested in different aspects of trading (technical, fundamental, educational,, sentiment, a mix of some or all, etc.). We Crawled 63 forex twitter accounts listed on the website and store it into `trader_account` list for future use.

```python
import requests
from bs4 import BeautifulSoup

headers = {'user-agent': 
           'Mozilla/5.0 (Macintosh Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36'}
url = "https://www.forexcrunch.com/60-top-forex-twitter-accounts/"
res = requests.get(url, headers=headers).text
soup = BeautifulSoup(res, "html.parser")
trader_account = []
table = soup.find(name="ol")
for account in table.find_all(name="li"):
    name = account.find(name="a").text
    name = name.replace("@", "")
    trader_account.append(name)
```


### Twint Variable Description
Here’s the full list of configuring options:

|Variable             |Type       |Description|
|---|---|---|
|Username             |(string) - |Twitter user's username|
|User_id              |(string) - |Twitter user's user_id|
|Search               |(string) - |Search terms|
|Geo                  |(string) - |Geo coordinates (lat,lon,km/mi.)|
|Location             |(bool)   - |Set to True to attempt to grab a Twitter user's location (slow).|
|Near                 |(string) - |Near a certain City (Example: london)|
|Lang                 |(string) - |Compatible language codes: https://github.com/twintproject/twint/wiki/Langauge-codes|
|Output               |(string) - |Name of the output file.|
|Elasticsearch        |(string) - |Elasticsearch instance|
|Timedelta            |(int)    - |Time interval for every request (days)|
|Year                 |(string) - |Filter Tweets before the specified year.|
|Since                |(string) - |Filter Tweets sent since date (Example: 2017-12-27).|
|Until                |(string) - |Filter Tweets sent until date (Example: 2017-12-27).|
|Email                |(bool)   - |Set to True to show Tweets that _might_ contain emails.|
|Phone                |(bool)   - |Set to True to show Tweets that _might_ contain phone numbers.|
|Verified             |(bool)   - |Set to True to only show Tweets by _verified_ users|
|Store_csv            |(bool)   - |Set to True to write as a csv file.|
|Store_json           |(bool)   - |Set to True to write as a json file.|
|Custom               |(dict)   - |Custom csv/json formatting (see below).|
|Show_hashtags        |(bool)   - |Set to True to show hashtags in the terminal output.|
|Limit                |(int)    - |Number of Tweets to pull (Increments of 20).|
|Count                |(bool)   - |Count the total number of Tweets fetched.|
|Stats                |(bool)   - |Set to True to show Tweet stats in the terminal output.|
|Database             |(string) - |Store Tweets in a sqlite3 database. Set this to the DB. (Example: twitter.db)|
|To                   |(string) - |Display Tweets tweeted _to_ the specified user.|
|All                  |(string) - |Display all Tweets associated with the mentioned user.|
|Debug                |(bool)   - |Store information in debug logs.|
|Format               |(string) - |Custom terminal output formatting.|
|Essid                |(string) - |Elasticsearch session ID.|
|User_full            |(bool)   - |Set to True to display full user information. By default, only usernames are shown.|
|Profile_full         |(bool)   - |Set to True to use a slow, but effective method to enumerate a user's Timeline.|
|Store_object         |(bool)   - |Store tweets/user infos/usernames in JSON objects.|
|Store_pandas         |(bool)   - |Save Tweets in a DataFrame (Pandas) file.|
|Pandas_type          |(string) - |Specify HDF5 or Pickle (HDF5 as default).|
|Pandas               |(bool)   - |Enable Pandas integration.|
|Index_tweets         |(string) - |Custom Elasticsearch Index name for Tweets (default: twinttweets).|
|Index_follow         |(string) - |Custom Elasticsearch Index name for Follows (default: twintgraph).|
|Index_users          |(string) - |Custom Elasticsearch Index name for Users (default: twintuser).|
|Index_type           |(string) - |Custom Elasticsearch Document type (default: items).|
|Retries_count        |(int)    - |Number of retries of requests (default: 10).|
|Resume               |(int)    - |Resume from a specific tweet id (**currently broken, January 11, 2019**).|
|Images               |(bool)   - |Display only Tweets with images.|
|Videos               |(bool)   - |Display only Tweets with videos.|
|Media                |(bool)   - |Display Tweets with only images or videos.|
|Replies              |(bool)   - |Display replies to a subject.|
|Pandas_clean         |(bool)   - |Automatically clean Pandas dataframe at every scrape.|
|Lowercase            |(bool)   - |Automatically convert uppercases in lowercases.|
|Pandas_au            |(bool)   - |Automatically update the Pandas dataframe at every scrape.|
|Proxy_host           |(string) - |Proxy hostname or IP.|
|Proxy_port           |(int)    - |Proxy port.|
|Proxy_type           |(string) - |Proxy type.|
|Tor_control_port     |(int) - Tor| control port.|
|Tor_control_password |(string) - |Tor control password (not hashed).|
|Retweets             |(bool)   - |Display replies to a subject.|
|Hide_output          |(bool)   - |Hide output.|
|Get_replies          |(bool)   - |All replies to the tweet.|

### Crawl Tweets
[Twint](https://github.com/twintproject/twint) is an advanced Twitter scraping tool written in Python that allows for scraping Tweets from Twitter profiles without using Twitter's API. We utilise twint to get tweets, and store the results into a pandas dataframe. We created a simple function that you can see in the actual project that integrate Pandas with Twint API for this part. Next, there are many features we have from the query we just did. There’s a lot of different things to do with this data, but for this project we’ll only use some of them, namely `date`, `time`, `username`, `tweet`, `hashtags`, `likes_count`, `replies_count`, and `retweets_count`.

```python
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# 63 Top Forex Twitter Accounts: 
# https://www.forexcrunch.com/60-top-forex-twitter-accounts/
# https://towardsdatascience.com/analyzing-tweets-with-nlp-in-minutes-with-spark-optimus-and-twint-a0c96084995f
def tweets_dateframe(search, output_file, year="2020"):
    # Configure
    c = twint.Config()
    c.Search = search
    c.Year = year
    c.Lang = "en"
    c.Pandas = True
    c.Store_csv = True
    c.Format = "Username: {username} |  Tweet: {tweet}"
    c.Output = output_file
    c.Hide_output = True
    # c.Limit = 10000
    # c.User_full = True
    # c.Since = since
    # c.Until = until

    # Run
    with HiddenPrints():
        print(twint.run.Search(c))
    
    return "Done scraping tweets!"

for year in tqdm(["2017", "2018", "2019", "2020"]):
    tweets_dateframe(search="FXstreetNews", output_file="forex.csv", year=year)

cols = ["date", "time", "username", "tweet", "hashtags", "likes_count", "replies_count", "retweets_count"]
df = pd.read_csv("forex.csv", usecols=cols)
print("# of tweets: {}".format(df.shape[0]))
df.sort_values(by="date", ascending=True, inplace=True)
df.reset_index(drop=True, inplace=True)
df.head()
```