import psycopg2
import configparser
import os
import csv


def build_hn_corpus():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Read the SQL query from the .pgsql file
    with open(os.path.join(script_dir, "hn_corpus.pgsql"), "r") as file:
        query = file.read()

    # Read database configuration from the .ini file
    config = configparser.ConfigParser()
    config.read(os.path.join(script_dir, "../database.ini"))

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

    # create sources directory if it doesn't exist
    if not os.path.exists(os.path.join(script_dir, "../sources")):
        os.makedirs(os.path.join(script_dir, "../sources"))

    # write the output to a csv
    with open(
        os.path.join(script_dir, "../sources/hn_corpus.csv"), "w", encoding="utf-8"
    ) as file:
        csvwriter = csv.writer(file)
        csvwriter.writerow(["title", "by", "time", "url", "score"])
        for row in rows:
            csvwriter.writerow(row)

    # write titles to a .txt file
    with open(
        os.path.join(script_dir, "../sources/hn_title_corpus.txt"),
        "w",
        encoding="utf-8",
    ) as file:
        for row in rows:
            file.write(f"{row[0]} ")


if __name__ == "__main__":
    build_hn_corpus()
