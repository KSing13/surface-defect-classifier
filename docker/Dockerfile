# Base image with Python 3.10
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements from parent folder
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY ../ .

# Default command to train model
CMD ["python", "src/train.py"]
