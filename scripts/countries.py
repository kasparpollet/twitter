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

def calculate_countries_per_week(df):
    new_df = pd.DataFrame()

    week = df.created_at.dt.week.tolist()
    year = df.created_at.dt.year.tolist()
    week_year = []
    
    for i in range(len(year)):
        week_year.append(f'{week[i]} {year[i]}')
    df['week'] = week_year

    countries = df.geo_location.unique().tolist()
    weeks = df.week.unique().tolist()
    for country in countries:
        for week in weeks:
            filtered = df[(df.geo_location == country) & (df.week == week)]
            if not filtered.empty:
                new_entry = {}
                new_entry['country'] = [country]
                new_entry['week'] = [week]
                new_entry['sentiment'] = [round(filtered.sentiment_compound.mean(),2)]

                labels = ['pos', 'neg', 'neu']
                total = len(filtered)
                new_entry['total_tweets'] = [total]
                for x in labels:
                    counting = len(filtered[filtered.label == x])
                    count = round(counting/total*100, 2)
                    new_entry[x] = [counting]
                    new_entry[x+'_per'] = [count]

                new_entry['diff'] = [round(new_entry['pos_per'][0] - new_entry['neg_per'][0], 2)]
                new_entry_df = pd.DataFrame.from_dict(new_entry)
                new_df = pd.concat([new_df, new_entry_df], axis=0)
    
    return new_df