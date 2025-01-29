from torch import nn
import torch
from torch.utils.data import Dataset


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context):
        embedded = self.embeddings(context)
        embedded_mean = torch.mean(embedded, dim=1)
        out = self.linear(embedded_mean)
        return out


class CBOWDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        context, target = self.pairs[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(
            target, dtype=torch.long
        )


def generate_cbow_pairs_from_tokens(tokeniser, corpus, window_size):
    padding_token = "<PAD>"
    training_data = []

    for line in corpus:
        padded_line = (
            [padding_token] * window_size
            + line.split(" ")
            + [padding_token] * window_size
        )

        token_ids = tokeniser.tokens_to_token_ids(padded_line)

        for i in range(window_size, len(token_ids) - window_size):
            target_token = token_ids[i]
            context_tokens = (
                token_ids[i - window_size : i] + token_ids[i + 1 : i + window_size + 1]
            )
            training_data.append((context_tokens, target_token))

    return CBOWDataset(training_data)
