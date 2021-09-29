# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 11:03:39 2021

@author: Jaime
"""

import pandas as pd
import seaborn as sns


import tweepy as tw

import re


import warnings
warnings.filterwarnings("ignore")

sns.set(font_scale=1.5)
sns.set_style("whitegrid")

def remove_url(txt):
    """Replace URLs found in a text string with nothing 
    (i.e. it will remove the URL from the string).

    Parameters
    ----------
    txt : string
        A text string that you want to parse and remove urls.

    Returns
    -------
    The same txt string with url's removed.
    """

    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())

consumer_key= '0JaPiWrKpeuN5XKrJ62QjhnOZ'
consumer_secret= '6CfnfH8y7PQUvz1eDKTZAjOf6WCgFYFbhriMQtecBROHyOVMFr'
access_token= '1438084578984476677-xH18IauzOG0abRlBDIu2ilPBxSV0y9'
access_token_secret= '7oasPw5UUlsIOArbDHGvoZASefAs5wF4LI5wN8Lur9Iml'

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

# Define the search term and the date_since date as variables
search_words = ['#afghanistan -filter:retweets', '#AfghanistanRefugees -filter:retweets', 
                '#AfghanistanCrisis -filter:retweets', '#PakistanIsTaliban -filter:retweets',
             '#TalibanTerror -filter:retweets', '#UNHCR -filter:retweets', '#humanitarianAssist -filter:retweets',
             '#AfganistanWomen -filter:retweets', '#SanctionPakistan -filter:retweets',
              '#unitednations -filter:retweets', '#AfghanistanBurning -filter:retweets', 
              '#Panjshir -filter:retweets', '#kaboel -filter:retweets', '#Humanrights -filter:retweets',
              '#NoToTaliban -filter:retweets',
              '#AfghanistanDisaster -filter:retweets',
              '#afghanrefugees -filter:retweets', '#TalibanTakeover -filter:retweets',
              '#AfghanRefugees -filter:retweets']


date_since = "2006-6-1"

# Collect tweets
twlist = []
for s in search_words:
    tweets = tw.Cursor(api.search_tweets,
                       q=s,
                       lang="en",
                       since=date_since).items(100)
    
    tweets_no_urls = [remove_url(tweet.text) for tweet in tweets]
    twlistap = [tweet for tweet in tweets_no_urls]
    twlist.extend(twlistap)



df = pd.DataFrame(data=twlist, 
                    columns=['tweet'])

print(df.head(1))
print(df.shape)

