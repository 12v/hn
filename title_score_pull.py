import psycopg2
import pandas as pd
import configparser
import numpy as np
from tokeniser import Tokeniser
from pathlib import Path


# Read database configuration from the .ini file
config = configparser.ConfigParser()
config.read("./database.ini")

db_params = {
    "dbname": config["postgresql"]["dbname"],
    "user": config["postgresql"]["user"],
    "password": config["postgresql"]["password"],
    "host": config["postgresql"]["host"],
    "port": config["postgresql"]["port"],
}

print("Config sections found:", config.sections())  # Should show ['postgresql']

with psycopg2.connect(**db_params) as conn:
    # Parameterized query
    query = """
        select id
    ,title
    ,score
    ,by as author
    from hacker_news.items 

    where type = 'story'
    and title is not null
    LIMIT 1000
    """

# Read directly into DataFrame 
df = pd.read_sql_query(query, conn)
    
# Optionally specify column dtypes
df = pd.read_sql_query(
        query,
        conn
    )

df.head()

# Initialize the Tokeniser
tokeniser = Tokeniser()

# Apply tokeniser._normalise_text() to each row in df.title
df['title'] = df['title'].apply(lambda x: ' '.join(tokeniser._normalise_text(x)))

# Split titles into words and create dictionary
df['words'] = df['title'].str.split()
df['count'] = df['title'].str.split().str.len()

