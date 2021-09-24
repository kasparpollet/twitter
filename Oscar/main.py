import pandas as pd
import os
from dotenv import load_dotenv

from twitter import TwitterApi
from database import DataBase
from unhcr import Unhcr

if __name__ == "__main__":
    # RUN CODE HERE
    load_dotenv()
    twitter = TwitterApi()
    tweet = twitter.get_id(457)
    #hashtag = twitter.get_hashtag()
    print(twitter.get_hashtag())

