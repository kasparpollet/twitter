from dotenv import load_dotenv

from scripts.twitter import TwitterApi
from scripts.database import DataBase
from scripts.unhcr import Unhcr
from scripts.clean import Clean

#Wordcloud imports
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt

def get_hashtags_from_file():
    with open('./files/hashtags.txt') as f:
        content = [line.split('\n')[0] for line in f.readlines()]
    return content

def display_wordcloud(df):
    unuseful_words = [word.replace('#', '').lower() for word in get_hashtags_from_file()]
    unuseful_words += ['https', 't', 'afghan', 'afghanistan', 'new', 'amp', 's']
    my_stopwords = ENGLISH_STOP_WORDS.union(unuseful_words)
    vect = CountVectorizer(lowercase = True, stop_words=my_stopwords)
    vect.fit(df.text)
    X = vect.transform(df.text)
    # Create and generate a word cloud image 
    my_cloud = WordCloud(background_color='white',stopwords=my_stopwords).generate(' '.join(df['text']))

    # Display the generated wordcloud image
    plt.imshow(my_cloud, interpolation='bilinear') 
    plt.axis("off")

    # Don't forget to show the final image
    plt.show()


def __init__():
    load_dotenv()
    twitter = TwitterApi()
    unhcr = Unhcr()
    db = DataBase()
    return twitter, unhcr, db

if __name__ == "__main__":
    # RUN CODE HERE
    twitter, unhcr, db = __init__()
    tweets = db.get_tweets()
    cleaned_tweets = Clean(tweets)
    
    matrix = cleaned_tweets.matrix
    print(matrix)
    # cleaned_tweets.display_wordcloud()

    # test = twitter.get_hashtags(get_hashtags_from_file(), 1)
    # display_wordcloud(tweet)
