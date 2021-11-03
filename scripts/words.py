
from numpy.lib.function_base import delete
import requests
import pandas as pd
import numpy as np
import seaborn as sns
#Natural language processing tool-kit       
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

# from main import get_hashtags_from_file

#Wordcloud imports
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
import matplotlib.pyplot as plt
# nltk.download("stopwords")


def display_wordcloud(self):
        from main import get_hashtags_from_file

        Mask = np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream=True).raw))
        image_colors = ImageColorGenerator(Mask)

        # Create and generate a word cloud image 
        my_cloud = WordCloud(background_color='black', mask=Mask).generate(' '.join(self.df['text']))

        # Display the generated wordcloud image
        plt.imshow(my_cloud.recolor(color_func=image_colors), interpolation='bilinear') 
        plt.axis("off")

        # Don't forget to show the final image
        plt.show()

def graph(word_frequency, sent):
    labels = word_frequency[0][1:51].index
    title = "Word Frequency for %s" %sent
    #Plot the figures
    plt.figure(figsize=(10,5))
    plt.bar(np.arange(50), word_frequency[0][1:51], width = 0.8, 
            color = sns.color_palette("bwr"), alpha=0.5, 
            edgecolor = "black", capsize=8, linewidth=1)
    plt.xticks(np.arange(50), labels, rotation=90, size=14)
    plt.xlabel("50 more frequent words", size=14)
    plt.ylabel("Frequency", size=14)
    #plt.title('Word Frequency for %s', size=18) %sent;
    plt.title(title, size=18)
    plt.grid(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.show() 