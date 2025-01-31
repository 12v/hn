FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY app-requirements.txt .


# Install torch first so it doesn't include CUDA
RUN pip install --no-cache-dir torch --index-url=https://download.pytorch.org/whl/cpu

# Install dependencies
RUN pip install --no-cache-dir -r app-requirements.txt

COPY weights-text8-hn.pt .
COPY vocab_mapping.txt .
COPY model_weights.pth .
COPY *.py .

# Expose the port the app runs on
EXPOSE 60606

# Command to run the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "60606"]
