from fastapi import FastAPI, Request
import embedding_inference
from neural_network import NeuralNetwork
import torch

app = FastAPI()

model = NeuralNetwork()

with open("model_weights.pth", "rb") as f:
    model.load_state_dict(torch.load(f, map_location=torch.device("cpu")))

model.eval()

print("starting up")


@app.post("/prediction")
async def predict(request: Request):
    body = await request.json()
    title = body["title"]
    print(title)

    embeddings = embedding_inference.text_to_embeddings(title)

    score = model.forward(embeddings.mean(dim=0))

    rounded_score = round(score.item())
    print(rounded_score)
    return {"score": rounded_score}

    # # return predicted score

    # # data = await request.json()
    # # Generate a random integer between 0 and 2500 with exponential decay
    # await asyncio.sleep(random.uniform(0, 5))

    # prediction = int(random.expovariate(1 / 20))
