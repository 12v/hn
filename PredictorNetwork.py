import random
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from numpy import array
from word2vec import Word2Vec
from tokeniser import Tokeniser
from embedding_inference import text_to_embeddings
import torch.nn.functional as F
from datasets import load_dataset
from neural_network import NeuralNetwork

##
##
##
# Import the CSV file containing HN titles, mean vectors and scores, normalised to to the maximum score in the dataset to get targets to always be between 0-1
df = pd.read_pickle("hn_embedded.pkl")
print("loaded in csv corpus")

# Print the word, tokens, and token IDs

##
##
##
# Create dictionary in matrix and score format
data = {"matrix": df["mean_embedding"].tolist(), "score": df["score"].values}
dataframe = pd.DataFrame(data)
print("created dataframe for loading")
##
##
##
# Generating random data to test training

# data = {
#     "matrix": [torch.rand(64) for _ in range(98735)],
#     "score": df['score'],
#     }


# dataframe = pd.DataFrame(data)
##
##
##


class ProcessedData(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Extract the matrix and score
        matrix = self.dataframe.iloc[idx]["matrix"]
        score = self.dataframe.iloc[idx]["score"]

        # Convert to PyTorch tensors
        matrix_tensor = torch.tensor(matrix, dtype=torch.float32).squeeze()
        score_tensor = torch.tensor(score, dtype=torch.float32)

        return matrix_tensor, score_tensor


# Create Dataset
dataset = ProcessedData(dataframe)

# Defining DataLoader
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


criterion = nn.MSELoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
##
##
##
epoch_losses = []

model = NeuralNetwork()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 30
for epoch in range(epochs):
    epoch_loss = 0.0
    # print("Training epoch= " +epoch)
    for batch in dataloader:
        # Get the input matrix and target score
        inputs, targets = batch

        # Forward pass
        predictions = model(inputs)
        loss = criterion(predictions.squeeze(), targets)  # Squeeze to match dimensions

        # Backward pass and optimisation
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        epoch_loss += loss.item()

    epoch_loss /= len(dataloader)
    epoch_losses.append(epoch_loss)

print("Training done!")


print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

epoch_losses = [round(loss, 4) for loss in epoch_losses]
##
##
##
# Sampling random chunks of dataframe to test accuracy of network
sampled_data = dataframe.sample(
    n=100, random_state=42
)  # n=100 to get 100 random samples

# Extract the input features (second column) and target scores (third column)
inputs = list(sampled_data.iloc[:, 0])  # Extracting the feature vectors
targets = torch.tensor(
    sampled_data.iloc[:, 1].values, dtype=torch.float32
)  # Extracting the actual scores

# Convert inputs to tensors if necessary
inputs_tensor = torch.stack([torch.tensor(input_vec) for input_vec in inputs])

# Pass the inputs through the trained model to get predictions
model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # No need to track gradients during inference
    predictions = model(inputs_tensor)

# Calculate the absolute difference between predictions and actual scores
differences = torch.abs(predictions.squeeze() - targets)

# Compute the average difference
average_difference = torch.mean(differences)
##
##
##
print(average_difference)
