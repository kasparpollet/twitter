import pandas as pd
from sqlalchemy import create_engine

class DataBase:
    
    def __init__(self):
        self.database_url = ''
        self.engine = self.get_engine()

    def get_reviews(self):
        return pd.read_sql("SELECT * FROM ???", con=self.engine)

    def upload_data(self, df, name):
        df.to_sql(name=name,con=self.engine,if_exists='fail',index=False,chunksize=1000) 

    def create_enigne(self):
        return create_engine(self.database_url)
