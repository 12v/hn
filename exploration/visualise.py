import psycopg2
import matplotlib.pyplot as plt
import configparser

# Read the SQL query from the .pgsql file
with open("query.pgsql", "r") as file:
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

# Prepare data for plotting
authors = [row[0] for row in rows]
avg_scores = [row[1] for row in rows]

# Plot the data
plt.figure(figsize=(10, 5))
plt.bar(authors, avg_scores, color="blue")
plt.xlabel("Authors")
plt.ylabel("Average Score")
plt.title("Average Story Scores by Author")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
