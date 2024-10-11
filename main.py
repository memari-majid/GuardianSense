# Multimodal Medical Emergency Detection Agent with LLaMA 3.2 via Ollama

import random
import time
import cv2
import numpy as np
import torch
import librosa
import warnings
warnings.filterwarnings("ignore")

# Import necessary libraries for Ollama integration
import requests
import json

# Import models for image and speech processing
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from PIL import Image

# -------------------------------
# 1. Initialize LLaMA 3.2 via Ollama
# -------------------------------

def initialize_llama_via_ollama():
    """
    Initializes the LLaMA 3.2 model via Ollama.
    """
    # Ollama runs a local server that we can send HTTP requests to.
    # Ensure that Ollama is installed and running.
    # We'll define a function to send prompts to the Ollama server.
    base_url = "http://localhost:11434"  # Default Ollama API port

    def llama_model(prompt):
        headers = {"Content-Type": "application/json"}
        data = {
            "model": "llama3.2",
            "prompt": prompt
        }
        response = requests.post(f"{base_url}/api/generate", headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            result = response.json()
            return result['response']
        else:
            print(f"Error communicating with Ollama: {response.status_code}")
            return ""

    return llama_model

# -------------------------------
# 2. Synthetic Data Generation
# -------------------------------

def generate_synthetic_data():
    """
    Generate synthetic physiological data.
    """
    data = {
        'heart_rate': random.randint(50, 150),  # bpm
        'oxygen_saturation': random.uniform(85, 100),  # %
        'blood_pressure_systolic': random.randint(90, 160),  # mmHg
        'blood_pressure_diastolic': random.randint(60, 100),  # mmHg
    }
    return data

# -------------------------------
# 3. Video Processing Function
# -------------------------------

def process_video(video_path):
    """
    Process video and extract features (e.g., detect falls).
    """
    # For simplicity, we'll simulate video processing
    fall_detected = random.choice([True, False])
    return fall_detected

# -------------------------------
# 4. Image Processing Function
# -------------------------------

def process_image(image_path):
    """
    Analyze the facial expression in an image.
    """
    try:
        # Simulate emotion detection
        emotions = ['happy', 'sad', 'angry', 'surprised', 'neutral']
        dominant_emotion = random.choice(emotions)
        return dominant_emotion
    except Exception as e:
        print(f"Error in image processing: {e}")
        return None

# -------------------------------
# 5. Speech Processing Function
# -------------------------------

def process_speech(audio_path):
    """
    Convert speech in an audio file to text using a pre-trained model.
    """
    try:
        # Load pre-trained model and tokenizer
        tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

        # Load audio
        speech, rate = librosa.load(audio_path, sr=16000)

        # Tokenize
        input_values = tokenizer(speech, return_tensors='pt', padding='longest').input_values

        # Perform inference
        logits = model(input_values).logits

        # Get predicted ids
        predicted_ids = torch.argmax(logits, dim=-1)

        # Decode the ids to text
        transcription = tokenizer.decode(predicted_ids[0])

        return transcription.lower()
    except Exception as e:
        print(f"Error in speech processing: {e}")
        return None

# -------------------------------
# 6. Text Processing Function with LLaMA 3.2
# -------------------------------

def process_text_llama(input_text, llama_model):
    """
    Analyze input text using the LLaMA 3.2 model via Ollama.
    """
    # Create a prompt
    prompt = f"Analyze the following patient statement for signs of medical emergency:\n\n\"{input_text}\"\n\nIs there an emergency? Provide a brief explanation."
    response = llama_model(prompt)
    return response.strip()

# -------------------------------
# 7. Data Fusion and Decision-Making
# -------------------------------

def data_fusion(physio_data, fall_detected, emotion, speech_text, text_analysis):
    """
    Fuse data from various sources and make a decision.
    """
    alerts = []

    # Check physiological data
    if physio_data['heart_rate'] > 120 or physio_data['heart_rate'] < 50:
        alerts.append('Abnormal heart rate detected.')
    if physio_data['oxygen_saturation'] < 90.0:
        alerts.append('Low oxygen saturation detected.')
    if physio_data['blood_pressure_systolic'] > 140 or physio_data['blood_pressure_systolic'] < 90:
        alerts.append('Abnormal blood pressure detected.')

    # Check fall detection
    if fall_detected:
        alerts.append('Fall detected.')

    # Check emotion
    if emotion in ['sad', 'angry']:
        alerts.append(f'Negative emotion detected: {emotion}.')

    # Check speech content using LLaMA 3.2
    if speech_text:
        speech_analysis = process_text_llama(speech_text, llama_model)
        if 'emergency' in speech_analysis.lower():
            alerts.append('Emergency detected in speech.')

    # Check text input analysis
    if 'emergency' in text_analysis.lower():
        alerts.append('Emergency detected in text input.')

    # Decision-making
    if alerts:
        decision = 'Medical emergency detected! Triggering alarm.'
    else:
        decision = 'No emergency detected.'

    return alerts, decision

# -------------------------------
# 8. Main Agent Function
# -------------------------------

def medical_emergency_agent():
    # Initialize LLaMA 3.2 via Ollama
    llama_model = initialize_llama_via_ollama()

    # Generate synthetic data
    physio_data = generate_synthetic_data()

    # Process video data
    video_path = 'path_to_video.mp4'  # Replace with actual video path
    fall_detected = process_video(video_path)

    # Process image data
    image_path = 'path_to_image.jpg'  # Replace with actual image path
    emotion = process_image(image_path)

    # Process speech data
    audio_path = 'path_to_audio.wav'  # Replace with actual audio path
    speech_text = process_speech(audio_path)

    # Analyze speech text using LLaMA 3.2
    if speech_text:
        speech_analysis = process_text_llama(speech_text, llama_model)
    else:
        speech_analysis = 'No speech input.'

    # Process text data (e.g., user input)
    input_text = "I have severe chest pain and shortness of breath."  # Replace with actual text input
    text_analysis = process_text_llama(input_text, llama_model)

    # Perform data fusion and make decision
    alerts, decision = data_fusion(
        physio_data, fall_detected, emotion, speech_text, text_analysis
    )

    # Print results
    print("Physiological Data:", physio_data)
    print("Fall Detected:", fall_detected)
    print("Detected Emotion:", emotion)
    print("Transcribed Speech:", speech_text)
    print("\nSpeech Analysis:", speech_analysis)
    print("\nText Analysis:", text_analysis)
    print("\nAlerts:")
    for alert in alerts:
        print("-", alert)
    print("\nDecision:", decision)

# -------------------------------
# 9. Run the Agent
# -------------------------------

if __name__ == "__main__":
    medical_emergency_agent()
