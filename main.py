from os import remove
from dotenv import load_dotenv
import re

from scripts.twitter import TwitterApi
from scripts.database import DataBase
from scripts.unhcr import Unhcr

#Natural language processing tool-kit
import nltk           
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

#Wordcloud imports
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt
from nltk import word_tokenize

def get_hashtags_from_file():
    with open('./files/hashtags.txt') as f:
        content = [line.split('\n')[0] for line in f.readlines()]
    return content


def clean_text(text):
    whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    cleanText = re.sub(r'^https?:\/\/.[\r\n]', '', text, flags=re.MULTILINE)
    cleanText = text.replace("<br>", " ")
    cleanText = cleanText.replace("\n", " ")
    cleanText = cleanText.encode('ascii', 'ignore').decode('ascii')
    return ''.join(filter(whitelist.__contains__, cleanText))


def remove_stopwords(tweet):
    # Create stopword list
    nltk.download("stopwords")
    stop = set(stopwords.words('english'))
    snow = nltk.stem.SnowballStemmer('english')
    for index, row in tweet.iterrows():
        words = [snow.stem(word) for word in row['text'].split() if word not in stop]
        tweet.at[index, 'text'] = words

def display_wordcloud(df):
    my_cloud1 = WordCloud(background_color='black').generate(' '.join(df['text']))
    # Display the generated wordcloud image
    plt.imshow(my_cloud1, interpolation='bilinear') 
    plt.axis("off")

    # Don't forget to show the final image
    plt.show()
    
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

def tokenizer(tweets):
    # Build the first vectorizer
    vect1 = CountVectorizer().fit(tweets.text)
    vect1.transform(tweets.text)

    # Build the second vectorizer
    vect2 = CountVectorizer(token_pattern=r'\b[^\d\W][^\d\W]').fit(tweets.text)
    vect2.transform(tweets.text)

    # Print out the length of each vectorizer
    print('Length of vectorizer 1: ', len(vect1.get_feature_names()))
    print('Length of vectorizer 2: ', len(vect2.get_feature_names()))

    # # Create a list of lists, containing the tokens from list_tweets
    # tokens = [word_tokenize(item) for item in tweets_list]

    # # Remove characters and digits , i.e. retain only letters
    # letters = [[word for word in item if word.isalpha()] for item in tokens]
    # # Remove characters, i.e. retain only letters and digits
    # let_digits = [[word for word in item if word.isalnum()] for item in tokens]
    # # Remove letters and characters, retain only digits
    # digits = [[word for word in item if word.isdigit()] for item in tokens]

    # # Print the last item in each list
    # print('Last item in alphabetic list: ', letters[2])
    # print('Last item in list of alphanumerics: ', let_digits[2])
    # print('Last item in the list of digits: ', digits[2])
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
    print(tweet.info())
    #print(tweet)
    #print(remove_stopwords(tweet))
    display_wordcloud(tweet)