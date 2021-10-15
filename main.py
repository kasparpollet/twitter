from dotenv import load_dotenv
import numpy as np
import requests

from scripts.twitter import TwitterApi
from scripts.database import DataBase
from scripts.unhcr import Unhcr
from scripts.clean import Clean

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


def __init__():
    load_dotenv()
    twitter = TwitterApi()
    unhcr = Unhcr()
    db = DataBase('newTweets')
    return twitter, unhcr, db

if __name__ == "__main__":
    # RUN CODE HERE
    twitter, unhcr, db = __init__()
    #tweets = db.get_tweets()
    #print(tweets)
    # cleaned_tweets = Clean(tweets)

    # matrix = cleaned_tweets.matrix
    # print(matrix)
    # cleaned_tweets.display_wordcloud()
    print(get_locations_from_file())

    test = twitter.get_hashtags(get_hashtags_from_file(), get_locations_from_file())
    print(test)
    #display_wordcloud_no_stopwords(tweets)

