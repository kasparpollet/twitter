from dotenv import load_dotenv

from scripts.twitter import TwitterApi
from scripts.database import DataBase
from scripts.unhcr import Unhcr

def get_hashtags_from_file():
    with open('./files/hashtags.txt') as f:
        content = [line.split('\n')[0] for line in f.readlines()]
    return content


def clean_text(text):
    whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    cleanText = text.replace("<br>", " ")
    cleanText = cleanText.replace("\n", " ")
    cleanText = cleanText.encode('ascii', 'ignore').decode('ascii')
    return ''.join(filter(whitelist.__contains__, cleanText))


def remove_stopwords(text):
    with open('./files/stopwords.txt') as f:
        for i in f.items():
            text = text.replace(i, '')
    return text


def __init__():
    load_dotenv()
    twitter = TwitterApi()
    unhcr = Unhcr()
    db = DataBase()
    return twitter, unhcr, db

if __name__ == "__main__":
    # RUN CODE HERE
    twitter, unhcr, db = __init__()

    # tweets_df = twitter.get_hashtags(get_hashtags_from_file())
    # db.upload_data(tweets_df, 'tweets', 'replace')

    #print(db.get_tweets()['text'].head(2))