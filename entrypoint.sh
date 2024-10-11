#!/bin/bash
set -e

# Start Ollama server in the background
ollama serve &

# Function to check if Ollama is up
wait_for_ollama() {
  local retries=30       # Number of attempts
  local wait=2           # Wait time between attempts in seconds

  echo "Waiting for Ollama server to start..."

  for ((i=1;i<=retries;i++)); do
    if ollama status >/dev/null 2>&1; then
      echo "Ollama is up and running."
      return 0
    fi
    echo "Attempt $i/$retries: Ollama not ready yet. Retrying in $wait seconds..."
    sleep $wait
  done

  echo "Error: Ollama server did not start within expected time."
  exit 1
}

# Call the function to wait for Ollama
wait_for_ollama

# Start Streamlit application
exec streamlit run main.py --server.port=8501 --server.address=0.0.0.0
