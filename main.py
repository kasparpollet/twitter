from dotenv import load_dotenv
import numpy as np
import time
import os
from scripts.twitter import TwitterApi
from scripts.database import DataBase
from scripts.unhcr import Unhcr
from scripts.clean import Clean
from scripts.words import graph, display_wordcloud
from scripts.classification_model import ClassificationModel, create_basic_models
from scripts.tryout.tr import t
from scripts.cluster import cluster


def get_hashtags_from_file():
    with open('./files/hashtags.txt') as f:
        content = [line.split('\n')[0] for line in f.readlines()]
    return content

def get_locations_from_file():
    with open('./files/locations.txt') as f:
        content = [line.split('\n')[0].split(';') for line in f.readlines()]
    return content

def get_sentiment(analyzer, texts):
    texts = texts.apply(lambda x: str(analyzer.polarity_scores(x)))
    return texts

def print_labels(df):
    print(df.label.value_counts()/13066)
    for i in range(18):
        print(i)
        print(df[df['cluster'] == i].label.value_counts())

def print_words(df):
    graph(df[(df.sentiment_neu > 0.5)], len=50, name='Neutral Tweats (>0.5)')
    graph(df[(df.sentiment_pos > 0.2)], len=50, name='Positive Tweats (>0.2)')
    graph(df[(df.sentiment_neg > 0.2)], len=50, name='Negative Tweats (>0.2)')

def do_sentiment(df, threshold=0.05):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import json

    analyzer = SentimentIntensityAnalyzer()
    df['sentiment'] = get_sentiment(analyzer, df['text'])
    df['sentiment_neg'] = df['sentiment'].apply(lambda x: json.loads(x.replace("'",'"'))['neg'])
    df['sentiment_neu'] = df['sentiment'].apply(lambda x: json.loads(x.replace("'",'"'))['neu'])
    df['sentiment_pos'] = df['sentiment'].apply(lambda x: json.loads(x.replace("'",'"'))['pos'])
    df['sentiment_compound'] = df['sentiment'].apply(lambda x: json.loads(x.replace("'",'"'))['compound'])
    df.drop('sentiment', axis=1, inplace=True)
    # df['sentiment'] = ''
    df['label'] = 'neu'
    df.loc[(df.sentiment_compound > threshold), 'label'] = 'pos'
    df.loc[(df.sentiment_compound < -1*threshold), 'label'] = 'neg'
    return df

def new_tweets(df):
    # Clean: rewrite the clean for new languages
    clean = Clean(df)
    df = clean.df
    
    # Do Sentiment on the data
    df = do_sentiment(df)

    # Cluster the tweets
    # df = cluster(df)
    
    # db.upload_data(df, 'finalTweets', error='extend')

    return df

def twitter_api(twitter, db):
    for hashtag in get_hashtags_from_file():
        print('getting:', hashtag)
        for geo, country in get_locations_from_file():
            print('getting:', country, hashtag)
            try:
                df = twitter.digital_ocean(geo, country, hashtag)
                db.upload_data(df, 'DigitalOceanTweetsNew', error='append')

                # Clean / filter / sentiment / upload tweets
                df = df[df['language'] == 'en']
                df = new_tweets(df)
                db.upload_data(df, 'finalTweets', error='append')

                # Drop duplicates
                final_df = db.get_tweets()
                final_df.drop_duplicates(subset=["text"],inplace=True)
                # final_df.reset_index(inplace=True, drop=True)
                db.upload_data(final_df, 'finalTweets', error='replace')

                print('sleeping for 15 minutes...')
                time.sleep(900)
            except Exception as e:
                print(e)
                print('something went wrong')

def __init__():
    load_dotenv()
    twitter = TwitterApi()
    unhcr = Unhcr()
    db = DataBase('finalTweets')
    return twitter, unhcr, db


if __name__ == "__main__":
    # RUN CODE HERE
    twitter, unhcr, db = __init__()
    df = db.get_tweets()
    # print(df)

    # df = df[df['language'] == 'en']
    # df.drop_duplicates(subset=["text"],inplace=True)
    # df = new_tweets(df)
    # print(df)
    # countries = df.geo_location.unique()
    # print(df.geo_location.value_counts())

    # for i in countries:
    #     # print(i)
    #     print(i, round(df[df['geo_location']==i].sentiment_compound.mean(), 2))

    # from scripts.countries import calculate_countries_sentiment
    from scripts.countries import calculate_countries_per_week
    country_sentiment_per_week = calculate_countries_per_week(df)
    db.upload_data(country_sentiment_per_week, name='countrySentimentPerWeek', error='replace')

    # calculate_countries_sentiment(df,db)

    # db = DataBase('tweets')

    #df = df[df['language'] == 'en']
    # create_basic_models(df)
    # twitter_api(twitter, db)

    # clean = Clean(df)
    # df = clean.df
    
    # df = do_sentiment(df)
    # print(df)

    # db.upload_data(df, 'tweetsBackup', error='replace')

