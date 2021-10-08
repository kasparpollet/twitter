from dotenv import load_dotenv
import datetime

from scripts.twitter import TwitterApi, TweetCriteria, TweetManager
from scripts.database import DataBase
from scripts.unhcr import Unhcr

def get_hashtags_from_file():
    with open('./files/hashtags.txt') as f:
        content = [line.split('\n')[0] for line in f.readlines()]
    return content

def remove_emojis(text):
    return text.encode('ascii', 'ignore').decode('ascii')


def __init__():
    load_dotenv()
    twitter = TwitterApi()
    unhcr = Unhcr()
    db = DataBase()
    return twitter, unhcr, db

if __name__ == "__main__":
    # RUN CODE HERE
    twitter, unhcr, db = __init__()

    # id = 1440495235587457024
    # # id = 1443970445695463426
    # date = datetime.datetime.now()
    # print(id)
    # while True:
    #     try:
    #         # date = datetime.datetime.timestamp(date)
    #         tweets_df = twitter.get_hashtags(['#taliban'], date)
    #         db.upload_data(tweets_df, 'test', 'append')
    #         # date = min(tweets_df['created_at']).date()
    #         # print(date)
    #     except Exception as e:
    #         print(e)


    # while True:
    #     try:
    #         tweets_df = twitter.get_hashtags(['#taliban'], id)
    #         db.upload_data(tweets_df, 'test', 'append')
    #         id = min(tweets_df['id']) - 1
    #         print(id)
    #     except Exception as e:
    #         print(e)


    # print(tweets_df['text'][3])
    # db.upload_data(tweets_df, 'tweets', 'replace')

    # print(db.get_tweets()['text'][3])

    tweetCriteria = TweetCriteria().setQuerySearch('taliban').setSince("2020-05-01").setUntil("2020-09-30").setMaxTweets(1)
    tweet = TweetManager.getTweets(tweetCriteria)[0]
        
    print(tweet.text)
