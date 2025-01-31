from fastapi import FastAPI, Request
import random

app = FastAPI()


@app.post("/prediction")
async def predict(request: Request):
    # data = await request.json()
    # Generate a random integer between 0 and 2500 with exponential decay
    prediction = int(random.expovariate(1 / 20))
    return {"score": prediction}
