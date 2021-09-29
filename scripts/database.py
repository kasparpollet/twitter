import pandas as pd
import os
from sqlalchemy import create_engine

class DataBase:
    
    def __init__(self):
        self.engine = create_engine(os.getenv('DATABASE_URL'))

    def get_unhcr(self):
        return pd.read_sql("SELECT * FROM unhcr", con=self.engine)

    def upload_data(self, df, name, error='fail'):
        df.to_sql(name=name,con=self.engine,if_exists=error,index=False,chunksize=1000) 
