from os import remove
from dotenv import load_dotenv

from scripts.twitter import TwitterApi
from scripts.database import DataBase
from scripts.unhcr import Unhcr

#Natural language processing tool-kit
import nltk           
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

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


def remove_stopwords(tweet):
    # Create stopword list
    nltk.download("stopwords")
    stop = set(stopwords.words('english'))
    temp =[]
    snow = nltk.stem.SnowballStemmer('english')
    for index, row in tweet.iterrows():
        print(tweet['text'])
        words = [snow.stem(word) for word in row['text'].split() if word not in stop]
        temp.append(words)
        tweet.at[index, 'text'] = words
        print(tweet['text'])
    return temp


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
    newestId = db.get_new_id()
    oldestId = db.get_old_id()
    tweet = db.get_tweets()
    #print(tweet)
    print(remove_stopwords(tweet))