import requests
import os
import tweepy
import pandas as pd
import json
import datetime
import csv
import time



class TwitterApi:
    
    def __init__(self):
        self.base_url = os.getenv('TWITTER_API_URL')
        self.token = os.getenv('TWITTER_BEARER_TOKEN')

    def get_hashtag(self):
        search_words = ['#afghanistan -filter:retweets', '#AfghanistanRefugees -filter:retweets',
                        '#AfghanistanCrisis -filter:retweets', '#PakistanIsTaliban -filter:retweets',
                        '#TalibanTerror -filter:retweets', '#UNHCR -filter:retweets',
                        '#humanitarianAssist -filter:retweets',
                        '#AfganistanWomen -filter:retweets', '#SanctionPakistan -filter:retweets',
                        '#unitednations -filter:retweets', '#AfghanistanBurning -filter:retweets',
                        '#Panjshir -filter:retweets', '#kaboel -filter:retweets', '#Humanrights -filter:retweets',
                        '#NoToTaliban -filter:retweets',
                        '#AfghanistanDisaster -filter:retweets',
                        '#afghanrefugees -filter:retweets', '#TalibanTakeover -filter:retweets',
                        '#AfghanRefugees -filter:retweets']
        
        # Keys and access for Twitter
        consumer_key = '0JaPiWrKpeuN5XKrJ62QjhnOZ'
        consumer_secret = '6CfnfH8y7PQUvz1eDKTZAjOf6WCgFYFbhriMQtecBROHyOVMFr'
        access_token = '1438084578984476677-xH18IauzOG0abRlBDIu2ilPBxSV0y9'
        access_token_secret = '7oasPw5UUlsIOArbDHGvoZASefAs5wF4LI5wN8Lur9Iml'
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        datesince = '2020-01-01'
        MAX_TWEETS = 10

        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        api = tweepy.API(auth)
        tweets = tweepy.Cursor(api.search_tweets, q='#Taliban', lang='en',
                               since=datesince).items(MAX_TWEETS)
        data = pd.DataFrame(data=[tweet for tweet in tweets], columns=['Tweets'])
        print(data)
        data.to_csv('tweets.csv')


        return

    def get_id(self, id):
        headers = {
            'Authorization': f'Bearer {self.token}',
        }

        return requests.get(self.base_url + str(id), headers=headers).text
