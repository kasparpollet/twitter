import re
import pandas as pd

#Natural language processing tool-kit
import nltk           
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

# from main import get_hashtags_from_file

#Wordcloud imports
from wordcloud import WordCloud
import matplotlib.pyplot as plt


class Clean:
    def __init__(self, df):
        self.df = self.__clean(df)
        self.remove_stopwords()
        self.matrix = self.tokenize()

    def __clean(self, df):
        df['text'] = df['text'].apply(lambda x: self.__clean_text(str(x)))
        return df

    def __clean_text(self, text):
        whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        clean_text = re.sub(r'^https?:\/\/.[\r\n]', '', text, flags=re.MULTILINE)
        clean_text = text.replace("<br>", " ")
        clean_text = clean_text.replace("\n", " ")
        clean_text = clean_text.encode('ascii', 'ignore').decode('ascii')
        clean_text = ''.join(i + ' ' for i in clean_text.split() if not i.startswith('http'))
        return ''.join(filter(whitelist.__contains__, clean_text))

    def remove_stopwords(self):
        # Create stopword list
        nltk.download("stopwords")
        stop = set(stopwords.words('english'))
        snow = nltk.stem.SnowballStemmer('english')

        for index, row in self.df.iterrows():
            words = ''.join(i + ' ' for i in [snow.stem(word) for word in row['text'].split() if word not in stop])
            self.df.at[index, 'text'] = words
        return self

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

