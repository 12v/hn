import os
from tokeniser import Tokeniser
import urllib.request
import torch
import wandb
from dataset import generate_pairs_from_tokens
import tqdm
from word2vec import Word2Vec

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


window_size = 1
batch_size = 512
embedding_dim = 64
epochs = 3
# arch = "cbow"
arch = "skipgram"
initial_lr = 0.001


tokeniser = Tokeniser()

vocab_size = len(tokeniser.vocab_mapping)

torch.manual_seed(1989)


def train_embeddings(corpus_path):
    with open(os.path.join(script_dir, corpus_path), "r") as f:
        corpus = f.read()

    dataset = generate_pairs_from_tokens(tokeniser, corpus.split("\n"), window_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    wandb.init(
        project="mlx6-word2vec",
        config={
            "learning_rate": initial_lr,
            "architecture": arch,
            "dataset": corpus_path,
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
            skipgram_rand = torch.randint(0, vocab_size, (context.size(0), 2)).to(
                device
            )

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


train_embeddings("sources/normalised_corpus_1.txt")
train_embeddings("sources/normalised_corpus_2.txt")
