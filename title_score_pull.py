import psycopg2
import pandas as pd
import configparser
import numpy as np
import torch
from tokeniser import Tokeniser
from word2vec import Word2Vec
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
# Apply tokeniser.text_to_token_ids() to each row in df.words
df['token_ids'] = df['words'].apply(lambda x: tokeniser.tokens_to_token_ids(x))
# Add a count of the number of token ids in each title
df['count'] = df['title'].str.split().str.len()


# Set up Word2Vec model parameters CHANGE TO MATCH YOUR weights.pt input
arch = "skipgram"
embedding_dim = 256
vocab_size = len(tokeniser.vocab_mapping)

# Initialize the model
model = Word2Vec(arch=arch, voc=vocab_size, emb=embedding_dim)

# Load the saved weights
with open("weights.pt", "rb") as f:
    model.load_state_dict(
        torch.load(f, map_location=torch.device("cpu"), weights_only=True)
    )

# Set the model to evaluation mode
model.eval()

# Get embedding weights matrix (vocab_size x embedding_dim)
embedding_weights = model.embeddings.weight.detach().cpu().numpy()


# Modified embedding processing
def get_embeddings(token_ids):
    embeddings = []
    for i in token_ids:
        # Handle both single numbers and arrays
        if isinstance(i, (list, np.ndarray)):
            # Flatten nested structures
            for sub_i in i:
                if sub_i < len(embedding_weights):
                    embeddings.append(embedding_weights[sub_i])
        elif i < len(embedding_weights):
            embeddings.append(embedding_weights[i])
    return np.array(embeddings)

# add embeddings to df
df['embeddings'] = df['token_ids'].apply(get_embeddings)

# Drop any rows with null token_ids
# df = df.dropna(subset=['token_ids']).replace('', np.nan).dropna()
# Example output
print(df[['title', 'embeddings']].head(2))

# Check for rows with 0 count
no_token_ids_rows = df[df['token_ids'].apply(len) == 0]
if not no_token_ids_rows.empty:
    print("Rows with 0 tokens found:")
    print(no_token_ids_rows)
else:
    print("No rows with 0 tokens found.")

# # Print the shape of each dict in df.embeddings
df['embedding_count'] = df['embeddings'].apply(lambda x: len(x))

def calculate_mean_embeddings(embedding_list):
    if len(embedding_list) == 0:
        return np.zeros(64)  # Match your embedding dimension
    
    # Convert list of arrays to 2D array
    embeddings_stack = np.stack(embedding_list)
    
    # Calculate mean along the first axis (word embeddings)
    return np.mean(embeddings_stack, axis=0)

# Apply to each row
df['mean_embedding'] = df['embeddings'].apply(calculate_mean_embeddings)


df['mean_embedding_count'] = df['mean_embedding'].apply(lambda x: len(x))

# Keep only the specified columns
df = df[['id', 'title', 'score', 'author', 'mean_embedding']]

print(df.head(2))


