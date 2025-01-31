import torch
from word2vec import Word2Vec
from tokeniser import Tokeniser
import torch.nn.functional as F

# Initialize the Tokeniser
tokeniser = Tokeniser()

# Set up Word2Vec model parameters
arch = "skipgram"
embedding_dim = 64
vocab_size = len(tokeniser.vocab_mapping)

# Initialize the model
model = Word2Vec(arch=arch, voc=vocab_size, emb=embedding_dim)

# Load the saved weights
with open("weights-text8-hn.pt", "rb") as f:
    model.load_state_dict(
        torch.load(f, map_location=torch.device("cpu"), weights_only=True)
    )

# Set the model to evaluation mode
model.eval()


def text_to_embeddings(text):
    token_ids = tokeniser.text_to_token_ids(text)
    tensor = torch.tensor(token_ids, dtype=torch.int64)
    embeddings = model.embeddings(tensor)
    return embeddings


if __name__ == "__main__":
    # Word you want to find the closest embedding for
    word = "jeff"

    # Tokenize the word and get its token ID
    token = tokeniser.text_to_tokens(word)
    token_id = tokeniser.text_to_token_ids(word)

    # Print the word, tokens, and token IDs
    print(f"Word: {word}, Token: {token}, Token ID: {token_id}")

    target_embedding = text_to_embeddings(word)
    print(target_embedding)

    # Get all embeddings (i.e., all word vectors in the vocabulary)
    all_embeddings = model.embeddings.weight

    # Compute cosine similarities between the target word embedding and all other embeddings
    cosine_similarities = F.cosine_similarity(target_embedding, all_embeddings, dim=1)

    # Exclude the input word by setting its similarity to -inf
    cosine_similarities[token_id] = -torch.inf

    # Find the index of the word with the highest cosine similarity
    most_similar_idx = torch.argmax(cosine_similarities)

    # Convert the closest token ID back to a word using tokeniser
    closest_token = tokeniser.token_ids_to_text([most_similar_idx.item()])

    # Print the result
    print(f"The closest word to '{word}' is: '{closest_token}'")
