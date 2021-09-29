#!/usr/bin/python3
from dotenv import load_dotenv

from scripts.twitter import TwitterApi
from scripts.database import DataBase
from scripts.unhcr import Unhcr


def __init__():
    load_dotenv()
    twitter = TwitterApi()
    unhcr = Unhcr()
    db = DataBase()
    return twitter, unhcr, db

if __name__ == "__main__":
    # RUN CODE HERE
    twitter, unhcr, db = __init__()

    reviews = DataBase().get_unhcr()
    print(reviews)
