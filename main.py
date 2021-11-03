from dotenv import load_dotenv
import numpy as np
import requests

from scripts.twitter import TwitterApi
from scripts.database import DataBase
from scripts.unhcr import Unhcr
from scripts.clean import Clean
from scripts.words import graph
from scripts.tryout.tr import t
#Wordcloud imports
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt

def get_hashtags_from_file():
    with open('./files/hashtags.txt') as f:
        content = [line.split('\n')[0] for line in f.readlines()]
    return content

def get_locations_from_file():
    with open('./files/locations.txt') as f:
        content = [line.split('\n')[0].split(';') for line in f.readlines()]
    return content

def display_wordcloud_no_stopwords(df):
    # Create and generate a word cloud image 
    Mask = np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream=True).raw))
    image_colors = ImageColorGenerator(Mask)

    # Create and generate a word cloud image 
    my_cloud = WordCloud(background_color='black', mask=Mask).generate(' '.join(df['text']))

    # Display the generated wordcloud image
    plt.imshow(my_cloud.recolor(color_func=image_colors), interpolation='bilinear') 
    plt.axis("off")

    # Don't forget to show the final image
    plt.show()

def get_sentiment(analyzer, texts):
    texts = texts.apply(lambda x: str(analyzer.polarity_scores(x)))
    return texts

def do_sentiment(df):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import json

    analyzer = SentimentIntensityAnalyzer()
    df['sentiment'] = get_sentiment(analyzer, df['text'])
    df['sentiment_neg'] = df['sentiment'].apply(lambda x: json.loads(x.replace("'",'"'))['neg'])
    df['sentiment_neu'] = df['sentiment'].apply(lambda x: json.loads(x.replace("'",'"'))['neu'])
    df['sentiment_pos'] = df['sentiment'].apply(lambda x: json.loads(x.replace("'",'"'))['pos'])
    df['sentiment_compound'] = df['sentiment'].apply(lambda x: json.loads(x.replace("'",'"'))['compound'])
    df.drop('sentiment', axis=1, inplace=True)
    return df

def __init__():
    load_dotenv()
    twitter = TwitterApi()
    unhcr = Unhcr()
    db = DataBase('CleanedDataNew')
    return twitter, unhcr, db

if __name__ == "__main__":
    # RUN CODE HERE
    twitter, unhcr, db = __init__()
    df = db.get_tweets()
    # print(tweets['text'].apply(lambda x: print(x)))
    # print(cleaned_tweets.df)
    # db.upload_data(cleaned_tweets.df, 'CleanedData', error='replace')
    # poep = tweets.text.tolist()
    # test(poep[0])
    # tweets['text'].apply(lambda x: test(str(x)))

    # matrix = cleaned_tweets.matrix
    # print(matrix)
    # cleaned_tweets.display_wordcloud()
    # print(get_locations_from_file())

    # clean = Clean(df)
    # df = clean.df
    
    df = do_sentiment(df)
    print(df)
    # db.upload_data(df, 'TestSentiment', error='replace')

