import pandas as pd
import re
#Natural language processing tool-kit
import nltk           
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

# from main import get_hashtags_from_file

#Wordcloud imports
from wordcloud import WordCloud
import matplotlib.pyplot as plt


class Clean:
    def __init__(self, df):
        self.df = df
        self.matrix = self.tokenize()

    def clean(self):
        self.df['text'] = self.df['text'].apply(lambda x: self.clean_text(str(x)))
        print(self.df['text'][1])

    def clean_text(self, text):
        whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        cleanText = text.replace("<br>", " ")
        cleanText = cleanText.replace("\n", " ")
        cleanText = cleanText.encode('ascii', 'ignore').decode('ascii')
        return ''.join(filter(whitelist.__contains__, text))

    def remove_stopwords(self, tweet):
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

    def tokenize(self):
        vec = CountVectorizer(lowercase=True, stop_words='english')
        wordcount = vec.fit_transform(self.df['text'].apply(lambda x: self.clean_text(str(x))).tolist())
        tokens = vec.get_feature_names()
        print(tokens)
        matrix = self.dtm(wordcount, tokens)
        return matrix

    def dtm(self, wm, feat_names):
        # create an index for each row
        doc_names = ['Doc{:d}'.format(idx) for idx, _ in enumerate(wm)]
        self.df = pd.DataFrame(data=wm.toarray(), index=doc_names,
                        columns=feat_names)

    def display_wordcloud(self):
        unuseful_words = [word.replace('#', '').lower() for word in get_hashtags_from_file()]
        unuseful_words += ['https', 't', 'afghan', 'afghanistan', 'new', 'amp', 's']
        my_stopwords = ENGLISH_STOP_WORDS.union(unuseful_words)
        vect = CountVectorizer(lowercase = True, stop_words=my_stopwords)
        vect.fit(self.df.text)
        X = vect.transform(self.df.text)
        # Create and generate a word cloud image 
        my_cloud = WordCloud(background_color='white',stopwords=my_stopwords).generate(' '.join(self.df['text']))

        # Display the generated wordcloud image
        plt.imshow(my_cloud, interpolation='bilinear') 
        plt.axis("off")

        # Don't forget to show the final image
        plt.show()

