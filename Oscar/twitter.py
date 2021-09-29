import requests
import os
import tweepy


class TwitterApi:
    
    def __init__(self):
        self.base_url = os.getenv('TWITTER_API_URL')
        self.token = os.getenv('TWITTER_BEARER_TOKEN')

    def get_hashtag(self):
        hashtags = []
        headers = {
            'Authorization': f'Bearer {self.token}',
        }
        api = requests.get(self.base_url + "search/recent?query=%40Taliban", headers=headers).text
        df = 
        return

    def get_id(self, id):
        headers = {
            'Authorization': f'Bearer {self.token}',
        }

        return requests.get(self.base_url + str(id), headers=headers).text
