# Use an official Python runtime as a base image
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    curl \
    && rm -rf /var/lib/apt/lists/*

<<<<<<< HEAD
=======
# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

>>>>>>> ac3e1c2 (update)
# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file to the working directory
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

<<<<<<< HEAD
# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh


# Expose necessary ports
EXPOSE 8501 11434

# Copy the entrypoint script into the container
COPY entrypoint.sh /entrypoint.sh

# Make the entrypoint script executable
RUN chmod +x /entrypoint.sh

# Define the entrypoint
ENTRYPOINT ["/entrypoint.sh"]
>>>>>>> ac3e1c2 (update)
