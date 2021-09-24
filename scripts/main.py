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
    tweet = twitter.get_id(475)
<<<<<<< HEAD
    hashtag = twitter.get_hashtag()
    print(hashtag)

=======
    print(tweet)
    unhcr = Unhcr()
    unhcr.exploration()
>>>>>>> 4a02dd3aa8ecb53d7ad357a5a796364cf235345d
