FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY app/requirements.txt .
COPY app/main.py .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 60606

# Command to run the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "60606"]
