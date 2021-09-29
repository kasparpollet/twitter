import requests
import os
# import tweepy


class TwitterApi:
    #https://api.twitter.com/2/tweets/search/recent?query=%23afghanistan&tweet.fields=created_at,text,lang,possibly_sensitive,in_reply_to_user_id
    def __init__(self):
        self.base_url = os.getenv('https://api.twitter.com/2/tweets/')
        self.token = os.getenv('TWITTER_BEARER_TOKEN')

    def get_hashtag(self):
        hashtags = []
        headers = {
            'Authorization': f'Bearer {self.token}',
        }
        api = requests.get(self.base_url + "search/recent?query=%40Taliban", headers=headers).text
        return api

    def get_id(self, id):
        headers = {
            'Authorization': f'Bearer {self.token}',
        }

        return requests.get(self.base_url + str(id), headers=headers).text
