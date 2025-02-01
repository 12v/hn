import psycopg2
import pandas as pd
import configparser
import numpy as np
import torch
import tqdm
import wandb
from tokeniser import Tokeniser
from word2vec import Word2Vec
from pathlib import Path
import embedding_inference


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

# print("Config sections found:", config.sections())  # Should show ['postgresql']

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

torch.manual_seed(2)


df['embeddings'] = df['title'].apply(embedding_inference.text_to_embeddings)
# Calculate the mean torch embedding for each row in df.embeddings
df['mean_embedding'] = df['embeddings'].apply(lambda x: torch.tensor(x).mean(dim=0))

# print(df['embeddings'].iloc[0])
# print(df['mean_embedding'].iloc[0])

print(df['title'].iloc[0])
# print(df['embeddings'].iloc[0].shape)
# print(df['mean_embedding'].iloc[0].shape)

# Define a simple neural network
class NeuralNetwork(torch.nn.Module):
    def __init__(self, emb):
        super(NeuralNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(emb, 32)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.2)
        self.fc2 = torch.nn.Linear(32, 4)
        self.relu2 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(p=0.1)
        self.fc3 = torch.nn.Linear(4, 1)

    def forward(self, inpt):
        out = self.fc1(inpt)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        return out


if torch.cuda.is_available():
        device = torch.device("cuda")
elif torch.backends.mps.is_available():
        device = torch.device("mps")
else:
        device = torch.device("cpu")
print("device being used", device)
# Initialize the neural network
embedding_dim = 64
predictor_model = NeuralNetwork(embedding_dim)
initial_lr = 0.0001
epochs = 1


# Define loss function and optimizer
criterion = torch.nn.MSELoss()
print("predictor_model", sum(p.numel() for p in predictor_model.parameters()))
optimiser = torch.optim.Adam(predictor_model.parameters(), lr=initial_lr)

# Prepare the data
input_tensor = torch.as_tensor(np.stack(df['mean_embedding']), dtype=torch.float32)
target_tensor = torch.tensor(df['score'].values, dtype=torch.float32).view(-1, 1)
dataset = torch.utils.data.TensorDataset(input_tensor, target_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True)

print(input_tensor.shape)
print(target_tensor.shape)

# Train the neural network
wandb.init(
        project="mlx6-predictor",
        config={
            "learning_rate": initial_lr,
            "embedding_dim": embedding_dim,
            "epochs": epochs,
        },
    )
predictor_model.to(device)
for epoch in range(epochs):
    prgs = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
    for inpt, trgs in prgs:
        inpt, trgs = inpt.to(device), trgs.to(device)
        optimiser.zero_grad()
        outputs = predictor_model(inpt)
        loss = criterion(outputs, trgs)
        loss.backward()
        optimiser.step()
        wandb.log({"loss": loss.item()})

print("Saving...")
torch.save(predictor_model.state_dict(), "./predictor-weights.pt")
print("Uploading...")
artifact = wandb.Artifact("predictor-weights", type="model")
artifact.add_file("./predictor-weights.pt")
wandb.log_artifact(artifact)
print("Done!")
wandb.finish()