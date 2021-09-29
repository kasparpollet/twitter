import requests
import os
import tweepy
import pandas as pd

class TwitterApi:
    
    def __init__(self):
        self.base_url = os.getenv('TWITTER_API_URL')
        self.api = self.__setup()

    def __setup(self):
        '''SEting up the twitter api'''
        consumer_key = os.getenv('TWITTER_KEY')
        consumer_secret = os.getenv('TWITTER_SECRET_SECRET_KEY')
        access_token = os.getenv('TWITTER_ACCESS_TOKEN')
        access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        return tweepy.API(auth)

    def get_hashtag(self, hashtag):
        datesince = '2020-01-01'
        MAX_TWEETS = 10

        tweets = tweepy.Cursor(self.api.search_tweets, q=f'#{hashtag}', lang='en',
                               since=datesince).items(MAX_TWEETS)
        data = pd.DataFrame(data=[tweet for tweet in tweets], columns=['Tweets'])
        return data

    def get_id(self, id):
        headers = {
            'Authorization': f'Bearer {self.token}',
        }

        return requests.get(self.base_url + str(id), headers=headers).text
