# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install git with retry mechanism
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python download_nltk_data.py

# Make port 8000 available to the world outside this container
EXPOSE ${PORT:-8000}

# Run main.py when the container launches
CMD ["python", "main.py"]
