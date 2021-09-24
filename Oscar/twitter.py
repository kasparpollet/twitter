import requests
import os
import tweepy

#hashtag = '#Afghanistan'

class TwitterApi:
    
    def __init__(self):
        self.base_url = os.getenv('TWITTER_API_URL')
        self.token = os.getenv('TWITTER_BEARER_TOKEN')

    def get_hashtag(self):
        headers = {
            'Authorization': f'Bearer {self.token}',
        }

        return requests.get(self.base_url + "search/recent?query=%40Taliban", headers=headers).text

    def get_id(self, id):
        headers = {
            'Authorization': f'Bearer {self.token}',
        }

        return requests.get(self.base_url + str(id), headers=headers).text

    def create_url(keyword, start_date, end_date, max_results=10):
        search_url = "https://api.twitter.com/2/tweets/search/tweets.json?q=#Afghanistan"  # Change to the endpoint you want to collect data from

        # change params based on the endpoint you are using
        query_params = {'query': keyword,
                        'start_time': start_date,
                        'end_time': end_date,
                        'max_results': max_results,
                        'expansions': 'author_id,in_reply_to_user_id,geo.place_id',
                        'tweet.fields': 'id,text,author_id,in_reply_to_user_id,geo,conversation_id,created_at,lang,public_metrics,referenced_tweets,reply_settings,source',
                        'user.fields': 'id,name,username,created_at,description,public_metrics,verified',
                        'place.fields': 'full_name,id,country,country_code,geo,name,place_type',
                        'next_token': {}}
        return (search_url, query_params)