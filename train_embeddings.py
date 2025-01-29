import os
from tokeniser.tokeniser import Tokeniser
import urllib.request

script_dir = os.path.dirname(os.path.abspath(__file__))

sources = {
    "text8": {
        "url": "https://huggingface.co/datasets/ardMLX/text8/resolve/main/text8",
        "path": os.path.join(script_dir, "sources/text8"),
    },
    "hn_title_corpus": {
        "url": "https://huggingface.co/datasets/12v12v/mlx6-1/resolve/main/hn_title_corpus.txt",
        "path": os.path.join(script_dir, "sources/hn_title_corpus.txt"),
    },
    "text8_normalised_corpus": {
        "url": "https://huggingface.co/datasets/12v12v/mlx6-1/resolve/main/normalised_corpus_1.txt",
        "path": os.path.join(script_dir, "sources/normalised_corpus_1.txt"),
    },
    "hn_title_normalised_corpus": {
        "url": "https://huggingface.co/datasets/12v12v/mlx6-1/resolve/main/normalised_corpus_2.txt",
        "path": os.path.join(script_dir, "sources/normalised_corpus_2.txt"),
    },
    "combined_normalised_corpus": {
        "url": "https://huggingface.co/datasets/12v12v/mlx6-1/resolve/main/normalised_corpuses.txt",
        "path": os.path.join(script_dir, "sources/normalised_corpuses.txt"),
    },
}

# Ensure the sources directory exists
os.makedirs(os.path.join(script_dir, "sources"), exist_ok=True)

# For each source, download the file if it doesn't exist
for name, source in sources.items():
    if not os.path.exists(source["path"]):
        print(f"{name} not found, downloading now...")
        urllib.request.urlretrieve(source["url"], source["path"])
        print(f"{name} downloaded and saved to {source['path']}")

tokeniser = Tokeniser()

text = "In this corpus there are words."

tokens = tokeniser.text_to_tokens(text)
token_ids = tokeniser.text_to_token_ids(text)
reconstructed_text = tokeniser.token_ids_to_text(token_ids)
print("Tokens:")
print(tokens)
print("Token IDs:")
print(token_ids)
print("Reconstructed text:")
print(reconstructed_text)

# To be continued!
