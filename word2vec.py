import torch
from torch import nn, optim
import os
from tokeniser import Tokeniser
from torch.utils.data import DataLoader
import time
import wandb
from cbow import CBOW, generate_cbow_pairs_from_tokens

window_size = 2
batch_size = 256
embedding_dim = 100


script_dir = os.path.dirname(__file__)

tokeniser = Tokeniser()

vocab_size = len(tokeniser.vocab_mapping)

torch.manual_seed(1989)

wandb.init(
    project="mlx6-word2vec",
    config={
        "learning_rate": 0.001,
        "architecture": "CBOW",
        "dataset": "text8",
        "epochs": 10,
    },
)


with open(os.path.join(script_dir, "sources/normalised_corpus_1.txt"), "r") as f:
    corpus = f.read()

# corpus = " ".join(
#     tokeniser.text_to_tokens(
#         "one two three four five one two three four five one two three four five one two three four five one two three four five one two three four five one two three four five one two three four five one two three four five one two three four five one two three four five one two three four five one two three four five one two three four five one two three four five one two three four five one two three four five one two three four five one two three four five one two three four five one two three four five one two three four five one two three four five one two three four five one two three four five one two three four five one two three four five one two three four five one two three four five one two three four five one two three four five one two three four five one two three four five one two three four five one two three four five one two three four five "
#     )
# )
dataset = generate_cbow_pairs_from_tokens(tokeniser, [corpus], window_size)
print("Training data generated")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = CBOW(vocab_size, embedding_dim)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model.to(device)
criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=0.001)


epochs = 10
for epoch in range(epochs):
    print(f"Epoch {epoch + 1} started")
    epoch_start_time = time.time()
    total_loss = 0
    total_batch_time = 0
    batch_count = 0

    for i, batch in enumerate(dataloader):
        batch_start_time = time.time()
        context, target = batch
        context, target = context.to(device), target.to(device)

        optimiser.zero_grad()
        logits = model(context)
        loss = criterion(logits, target)
        loss.backward()
        optimiser.step()

        total_loss += loss.item()

        batch_time = time.time() - batch_start_time
        total_batch_time += batch_time
        batch_count += 1

        # Print every 100 batches
        if (i + 1) % 100 == 0:
            avg_batch_time = total_batch_time / batch_count
            remaining_batches = len(dataloader) - (i + 1)
            print(
                f"\rBatches processed: {i + 1}/{len(dataloader)} | "
                f"Batches remaining: {remaining_batches} | "
                f"Avg time per batch: {avg_batch_time:.4f} sec | "
                f"Current loss: {loss.item():.4f}",
                end="",
            )

            wandb.log({"acc": loss.item(), "epoch": epoch + 1})

        # Final print for the epoch summary
    epoch_time = time.time() - epoch_start_time
    print(
        f"\nEpoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}, Time per epoch: {epoch_time:.2f} seconds"
    )


# embedding_dim = 50
# learning_rate = 0.001
# epochs = 100

# model = CBOW(len(tokeniser.vocab_mapping), embedding_dim)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# for epoch in range(epochs):
#     total_loss = 0
#     for context, target in training_data:
#         context_indices = torch.tensor(
#             tokeniser.tokens_to_token_ids(context), dtype=torch.long
#         )
#         target_index = torch.tensor(
#             tokeniser.text.to_token_ids(target), dtype=torch.long
#         )

#         outputs = model(context_indices)
#         loss = criterion(outputs, target_index)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     if (epoch + 1) % 10 == 0:
#         print(
#             f"Epoch {epoch + 1}, Total loss: {total_loss}, Loss: {total_loss / len(training_data)}"
#         )


# if __name__ == "__main__":
#     text = "word2vec is a popular model for word embeddings.  It is widely used in NLP"
