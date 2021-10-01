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

    def get_hashtags(self, hashtags):
        MAX_TWEETS = 2
        twlist = []

        try:
            for hashtag in hashtags:
                print('getting:', hashtag)
                tweets = tweepy.Cursor(self.api.search_tweets,
                                q=f'{hashtag} -filter:retweets',
                                lang="en").items(MAX_TWEETS)
                # tweets_no_urls = [remove_url(tweet.text) for tweet in tweets]
                twlistap = [tweet.text for tweet in tweets]
                twlist.extend(twlistap)
        except:
            pass
        
        print('finished')
        return pd.DataFrame(data=twlist, columns=['tweet'])        

    def get_id(self, id):
        headers = {
            'Authorization': f'Bearer {self.token}',
        }

        return requests.get(self.base_url + str(id), headers=headers).text
