import re
import urllib
from numpy.lib.function_base import delete
import requests
import pandas as pd
import numpy as np
import time

#Natural language processing tool-kit
import nltk           
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

# from main import get_hashtags_from_file

#Wordcloud imports
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
import matplotlib.pyplot as plt
# nltk.download("stopwords")


class Clean:
    def __init__(self, df):
        self.df = self.__clean(df)
        # self.remove_stopwords()
        # self.matrix = self.tokenize()

    def __clean(self, df):
        print('\nStart cleaning the dataframe...')
        start = time.time()

        new_tweets = []

        for index, tweet in df.iterrows():

            if tweet.language in ['en', 'es', 'de']:
                if tweet.language == 'en':
                    lang = 'english'
                if tweet.language == 'es':
                    lang = 'spanish'
                if tweet.language == 'de':
                    lang = 'german'
                stop_words = self.__get_stopwords(lang)
                stemmer = self.__get_stemmer(lang)
                tweet.text = self.__clean_text(tweet.text, stop_words, stemmer)
                new_tweets.append(list(tweet))

        cleaned_df = pd.DataFrame(new_tweets, columns=[col for col in df])

        # Remove empty reviews
        cleaned_df = cleaned_df.loc[lambda x: x['text'] != '']

        cleaned_df.reset_index(inplace=True, drop=True)

        end = time.time()
        print(f'Finished cleaning the dataframe in {end-start} seconds')
        return cleaned_df

    def __clean_text(self, text, stop_words, stemmer):
        whitelist = set('abcdefghijklmnopqrstuvwxyz# ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        clean_text = text.replace("<br>", " ")
        clean_text = clean_text.replace("\n", " ")
        clean_text = clean_text.encode('ascii', 'ignore').decode('ascii')
        clean_text = ''.join(i + ' ' for i in clean_text.split() if not i.startswith('http') and not i.startswith('@'))
        clean_text = ''.join(i + ' ' for i in [stemmer.stem(word) for word in clean_text.lower().split() if word not in stop_words])
        return ''.join(filter(whitelist.__contains__, clean_text))

    def __get_stopwords(self, language):
        """
        Cobine nltk's and hotel reviews specific stopwords and returns these as a set
        """
        stop_words = stopwords.words(language)
        return list(set(stop_words))

    def __get_stemmer(self, language):
        stemmer = nltk.stem.SnowballStemmer(language, ignore_stopwords=True)
        return stemmer

    def tokenize(self):
        vec = CountVectorizer(lowercase=True, stop_words='english')

        wordcount = vec.fit_transform(self.df['text'].tolist())
        tokens = vec.get_feature_names_out()
        matrix = self.__dtm(wordcount, tokens)
        return matrix

    def __dtm(self, wm, feat_names):
        # create an index for each row
        doc_names = ['Doc{:d}'.format(idx) for idx, _ in enumerate(wm)]
        matrix = pd.DataFrame(data=wm.toarray(), index=doc_names,
                        columns=feat_names)
        return matrix

    def display_wordcloud(self):
        from main import get_hashtags_from_file

        Mask = np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream=True).raw))
        image_colors = ImageColorGenerator(Mask)


        unuseful_words = [word.replace('#', '').lower() for word in get_hashtags_from_file()]
        unuseful_words += ['https', 't', 'afghan', 'afghanistan', 'new', 'amp', 's']
        my_stopwords = ENGLISH_STOP_WORDS.union(unuseful_words)

        # Create and generate a word cloud image 
        my_cloud = WordCloud(background_color='black',stopwords=my_stopwords, mask=Mask).generate(' '.join(self.df['text']))

        # Display the generated wordcloud image
        plt.imshow(my_cloud.recolor(color_func=image_colors), interpolation='bilinear') 
        plt.axis("off")

        # Don't forget to show the final image
        plt.show()

