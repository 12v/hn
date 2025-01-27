import psycopg2
import configparser
import re
import zipfile

# Read the SQL query from the .pgsql file
with open("title_corpus.pgsql", "r") as file:
    query = file.read()

# Read database configuration from the .ini file
config = configparser.ConfigParser()
config.read("../database.ini")

db_params = {
    "dbname": config["postgresql"]["dbname"],
    "user": config["postgresql"]["user"],
    "password": config["postgresql"]["password"],
    "host": config["postgresql"]["host"],
}

# Connect to your postgres DB
conn = psycopg2.connect(**db_params)

# Open a cursor to perform database operations
cur = conn.cursor()

# Execute the query
cur.execute(query)

# Retrieve query results
rows = cur.fetchall()

# Close communication with the database
cur.close()
conn.close()

with open("title_corpus.txt", "w") as file:
    for row in rows:
        clean_title = re.sub(r"\s+", " ", row[0].strip())
        file.write(f"{clean_title} ")

with zipfile.ZipFile("title_corpus.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
    zipf.write("title_corpus.txt", arcname="title_corpus.txt")
