from fastapi import FastAPI, Request
import random
import asyncio

app = FastAPI()


@app.post("/prediction")
async def predict(request: Request):
    # data = await request.json()
    # Generate a random integer between 0 and 2500 with exponential decay
    await asyncio.sleep(random.uniform(0, 5))

    prediction = int(random.expovariate(1 / 20))
    return {"score": prediction}
