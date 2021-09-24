import requests

class TwitterApi:
    
    def __init__(self, hashtag):
        self.call = hashtag
        self.url = ''
        # TODO get authentication from file

    def get_heshtag(self):
        pass