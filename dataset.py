import torch
import more_itertools


def generate_pairs_from_tokens(tokeniser, corpus, window_size):
    padding_token = "<PAD>"
    contexts = []
    centers = []

    for line in corpus:
        padded_line = (
            [padding_token] * window_size
            + line.split(" ")
            + [padding_token] * window_size
        )

        token_ids = tokeniser.tokens_to_token_ids(padded_line)

        windows = more_itertools.windowed(token_ids, 2 * window_size + 1)
        for window in windows:
            context = window[:window_size] + window[window_size + 1 :]
            center = window[window_size]
            contexts.append(context)
            centers.append(center)

    context_tensor = torch.LongTensor(contexts)
    center_tensor = torch.LongTensor(centers)
    return torch.utils.data.TensorDataset(center_tensor, context_tensor)
