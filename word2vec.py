import torch
from torch import nn, optim
import os
from tokeniser import Tokeniser
from torch.utils.data import DataLoader
import time
import wandb
from dataset import generate_pairs_from_tokens
import tqdm

script_dir = os.path.dirname(__file__)

window_size = 1
batch_size = 512
embedding_dim = 64
epochs = 10
arch = "cbow"
# arch = "skipgram"
initial_lr = 0.001


tokeniser = Tokeniser()

vocab_size = len(tokeniser.vocab_mapping)

torch.manual_seed(1989)

with open(os.path.join(script_dir, "sources/normalised_corpus_1.txt"), "r") as f:
    corpus = f.read()

dataset = generate_pairs_from_tokens(tokeniser, corpus.split("\n"), window_size)
dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class SkipGram(torch.nn.Module):
    def __init__(self, voc, emb):
        super().__init__()
        self.emb = torch.nn.Embedding(num_embeddings=voc, embedding_dim=emb)
        self.ffw = torch.nn.Linear(in_features=emb, out_features=voc, bias=False)
        self.sig = torch.nn.Sigmoid()

    def forward(self, inpt, trgs, rand):
        emb = self.emb(inpt)
        ctx = self.ffw.weight[trgs]
        rnd = self.ffw.weight[rand]
        out = torch.bmm(ctx, emb.unsqueeze(-1)).squeeze()
        rnd = torch.bmm(rnd, emb.unsqueeze(-1)).squeeze()
        out = self.sig(out)
        rnd = self.sig(rnd)
        pst = -out.log().mean()
        ngt = -(1 - rnd + 10 ** (-3)).log().mean()
        return pst + ngt


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


wandb.init(
    project="mlx6-word2vec",
    config={
        "learning_rate": initial_lr,
        "architecture": arch,
        "dataset": "text8",
        "epochs": epochs,
    },
)


if arch == "cbow":
    model = CBOW(vocab_size, embedding_dim)
elif arch == "skipgram":
    model = SkipGram(vocab_size, embedding_dim)
else:
    raise ValueError(f"Unknown architecture: {arch}")

print("param count:", sum(p.numel() for p in model.parameters()))

model.to(device)

optimiser = torch.optim.Adam(model.parameters(), lr=initial_lr)

for epoch in range(epochs):
    print(f"Epoch {epoch + 1} started")
    prgs = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)

    if arch == "cbow":
        criterion = nn.CrossEntropyLoss()

        for target, context in prgs:
            batch_start_time = time.time()
            context, target = context.to(device), target.to(device)

            optimiser.zero_grad()
            logits = model(context)
            loss = criterion(logits, target)
            loss.backward()
            optimiser.step()

            wandb.log({"loss": loss.item()})
    elif arch == "skipgram":
        for inpt, trgs in prgs:
            inpt, trgs = inpt.to(device), trgs.to(device)

            rand = torch.randint(0, len(words_to_ids), (inpt.size(0), 2)).to(device)
            optimiser.zero_grad()
            loss = model(inpt, trgs, rand)
            loss.backward()
            optimiser.step()
            wandb.log({"loss": loss.item()})
    else:
        raise ValueError(f"Unknown architecture: {arch}")


torch.save(model.state_dict(), os.path.join(script_dir, "weights.pt"))
print("Uploading...")
artifact = wandb.Artifact("model-weights", type="model")
artifact.add_file("./weights.pt")
wandb.log_artifact(artifact)
print("Done!")
wandb.finish()
