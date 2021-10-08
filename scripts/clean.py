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
        self.df = self.__clean(df)
        # self.matrix = self.tokenize()

    def __clean(self, df):
        df['text'] = df['text'].apply(lambda x: self.clean_text(str(x)))
        return df

    def clean_text(self, text):
        whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        cleanText = re.sub(r'^https?:\/\/.[\r\n]', '', text, flags=re.MULTILINE)
        cleanText = text.replace("<br>", " ")
        cleanText = cleanText.replace("\n", " ")
        cleanText = cleanText.encode('ascii', 'ignore').decode('ascii')
        cleanText = ''.join(i + ' ' for i in cleanText.split() if not i.startswith('http'))
        return ''.join(filter(whitelist.__contains__, cleanText))

    def remove_stopwords(self, tweet):
        # Create stopword list
        nltk.download("stopwords")
        stop = set(stopwords.words('english'))
        snow = nltk.stem.SnowballStemmer('english')
        for index, row in tweet.iterrows():
            words = [snow.stem(word) for word in row['text'].split() if word not in stop]
            tweet.at[index, 'text'] = words

    def tokenize(self):
        vec = CountVectorizer(lowercase=True, stop_words='english')
        wordcount = vec.fit_transform(self.df['text'].tolist())
        tokens = vec.get_feature_names_out()
        print(vec.get_feature_names())
        matrix = self.dtm(wordcount, tokens)
        return matrix

    def dtm(self, wm, feat_names):
        # create an index for each row
        doc_names = ['Doc{:d}'.format(idx) for idx, _ in enumerate(wm)]
        matrix = pd.DataFrame(data=wm.toarray(), index=doc_names,
                        columns=feat_names)
        return matrix

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

