import plotly.graph_objs as go
import dash
from dash.dcc.Graph import Graph
from dash.dependencies import Input, Output
from dash import dcc
from dash import html
from scripts.database import DataBase
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np


def calculate_countries_sentiment(data, db):
    df = data[['geo_location', 'label', 'sentiment_compound', 'text']]
    countries = df.geo_location.unique().tolist()
    new_df = pd.DataFrame({'country': countries})
    # sentiment = []
    for i in countries:
        sentiment = round(df.loc[df.geo_location==i].sentiment_compound.mean(),2)
        new_df.loc[new_df.country == i, 'sentiment'] = sentiment


        labels = ['pos', 'neg', 'neu']
        total = len(df[(df.geo_location == i)])
        for x in labels:
            counting = len(df[(df.geo_location == i) & (df.label == x)])
            count = round(counting/total*100, 2)
            new_df.loc[new_df.country == i, x] = counting
            new_df.loc[new_df.country == i, x+'_per'] = count


    new_df['diff'] = round(new_df['pos_per'] - new_df['neg_per'], 2)
    db.upload_data(new_df, name='countrySentiment', error='replace')
