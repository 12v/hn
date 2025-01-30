import torch
import os
from tokeniser import Tokeniser
import wandb
from dataset import generate_pairs_from_tokens
import tqdm

script_dir = os.path.dirname(__file__)

window_size = 1
batch_size = 512
embedding_dim = 64
epochs = 10
# arch = "cbow"
arch = "skipgram"
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


class Word2Vec(torch.nn.Module):
    def __init__(self, arch, voc, emb):
        super().__init__()
        self.embeddings = torch.nn.Embedding(num_embeddings=voc, embedding_dim=emb)
        self.linear = torch.nn.Linear(in_features=emb, out_features=voc, bias=False)
        self.sig = torch.nn.Sigmoid()

        if arch in ["cbow", "skipgram"]:
            self.arch = arch
        else:
            raise ValueError(f"Unknown architecture: {arch}")

    def forward(self, center, context, rand):
        input = center if self.arch == "skipgram" else context
        output = context if self.arch == "skipgram" else center

        input_embeddings = self.embeddings(input)
        output_weights = self.linear.weight[output]
        random_weights = self.linear.weight[rand]

        if self.arch == "skipgram":
            input_embeddings = input_embeddings.unsqueeze(-1)
        else:
            output_weights = output_weights.unsqueeze(1)
            random_weights = random_weights.unsqueeze(1)
            input_embeddings = input_embeddings.permute(0, 2, 1)

        dot_product = torch.bmm(output_weights, input_embeddings).squeeze()
        random_dot_product = torch.bmm(random_weights, input_embeddings).squeeze()

        activation = self.sig(dot_product)
        random_activation = self.sig(random_dot_product)

        positive_sampling_loss = -activation.log().mean()
        negative_sampling_loss = -(1 - random_activation + 10 ** (-3)).log().mean()
        return positive_sampling_loss + negative_sampling_loss


wandb.init(
    project="mlx6-word2vec",
    config={
        "learning_rate": initial_lr,
        "architecture": arch,
        "dataset": "text8",
        "epochs": epochs,
    },
)

model = Word2Vec(arch, vocab_size, embedding_dim)

print(arch)
print("param count:", sum(p.numel() for p in model.parameters()))

model.to(device)

optimiser = torch.optim.Adam(model.parameters(), lr=initial_lr)

for epoch in range(epochs):
    print(f"Epoch {epoch + 1} started")
    prgs = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)

    for center, context in prgs:
        center, context = center.to(device), context.to(device)

        cbow_rand = torch.randint(0, vocab_size, (center.size(0),)).to(device)
        skipgram_rand = torch.randint(0, vocab_size, (context.size(0), 2)).to(device)

        if arch == "cbow":
            rand = cbow_rand
        else:
            rand = skipgram_rand

        optimiser.zero_grad()
        loss = model(center, context, rand)

        loss.backward()
        optimiser.step()

        wandb.log({"loss": loss.item()})


torch.save(model.state_dict(), os.path.join(script_dir, "weights.pt"))
print("Uploading...")
artifact = wandb.Artifact("model-weights", type="model")
artifact.add_file("./weights.pt")
wandb.log_artifact(artifact)
print("Done!")
wandb.finish()
