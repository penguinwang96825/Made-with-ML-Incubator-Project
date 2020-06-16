# Made_with_ML_Incubator_Project
In this project, we focus on the researches applying natural language processing (NLP) technologies in the finance domain. 

## Import Packages
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

## Twint

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