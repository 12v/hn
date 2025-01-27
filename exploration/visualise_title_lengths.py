import psycopg2
import matplotlib.pyplot as plt
import configparser

# Read the SQL query from the .pgsql file
with open("visualise_title_lengths.pgsql", "r") as file:
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

# Calculate lengths of strings
lengths = [len(s[0]) for s in rows]

# Plot the histogram
plt.figure(figsize=(10, 5))
plt.hist(lengths, bins=range(1, max(lengths) + 2), edgecolor="black", alpha=0.7)
plt.xlabel("String Length")
plt.ylabel("Frequency")
plt.title("Length Distribution of Strings")
plt.xticks(range(1, max(lengths) + 1))
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
