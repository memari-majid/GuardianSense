# Multimodal Medical Emergency Detection Agent with Streamlit GUI and Comprehensive Error Handling

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
import os

# Import models for image and speech processing
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from PIL import Image

# Import Streamlit for web GUI
import streamlit as st

# -------------------------------
# 1. Initialize LLaMA via Ollama
# -------------------------------

def initialize_llama_via_ollama():
    """
    Initializes the LLaMA model via Ollama.
    """
    base_url = "http://localhost:11434"  # Default Ollama API port

    def llama_model(prompt):
        try:
            headers = {"Content-Type": "application/json"}
            data = {
                "model": "llama",
                "prompt": prompt
            }
            response = requests.post(
                f"{base_url}/api/generate",
                headers=headers,
                data=json.dumps(data),
                stream=True  # Enable streaming
            )
            if response.status_code == 200:
                output = ""
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        try:
                            json_data = json.loads(decoded_line)
                            output += json_data.get('response', '')
                        except json.JSONDecodeError:
                            continue
                return output.strip()
            else:
                print(f"Error communicating with Ollama: {response.status_code}")
                return ""
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with Ollama: {e}")
            return ""
        except Exception as e:
            print(f"Exception in llama_model: {e}")
            return ""
    return llama_model

# -------------------------------
# 2. Synthetic Data Generation
# -------------------------------

def generate_synthetic_data():
    """
    Generate synthetic physiological data mimicking Apple Watch Health data.
    """
    data = {
        'heart_rate': random.randint(50, 150),  # bpm
        'oxygen_saturation': random.uniform(85, 100),  # %
        'blood_pressure_systolic': random.randint(90, 160),  # mmHg
        'blood_pressure_diastolic': random.randint(60, 100),  # mmHg
        'steps': random.randint(0, 10000),  # Steps
        'calories_burned': random.uniform(0, 500),  # kcal
        'sleep_hours': random.uniform(0, 12),  # hours
    }
    return data

# -------------------------------
# 3. Video Processing Function
# -------------------------------

def process_video(video_path):
    """
    Process video and extract features (e.g., detect falls).
    """
    try:
        if not os.path.exists(video_path):
            print(f"Error: Video file {video_path} does not exist.")
            return False
        # For simplicity, we'll simulate video processing
        fall_detected = random.choice([True, False])
        return fall_detected
    except Exception as e:
        print(f"Error in video processing: {e}")
        return False

# -------------------------------
# 4. Image Processing Function
# -------------------------------

def process_image(image_path):
    """
    Analyze the facial expression in an image.
    """
    try:
        if not os.path.exists(image_path):
            print(f"Error: Image file {image_path} does not exist.")
            return None
        # Load image to ensure it exists and is readable
        img = Image.open(image_path)
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
        if not os.path.exists(audio_path):
            print(f"Error: Audio file {audio_path} does not exist.")
            return None
        # Load pre-trained model and processor
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

        # Load audio
        speech, rate = librosa.load(audio_path, sr=16000)

        if len(speech) == 0:
            print("Error: Audio file is empty or cannot be read.")
            return None

        # Tokenize
        input_values = processor(speech, return_tensors='pt', padding='longest').input_values

        # Perform inference
        logits = model(input_values).logits

        # Get predicted ids
        predicted_ids = torch.argmax(logits, dim=-1)

        # Decode the ids to text
        transcription = processor.decode(predicted_ids[0])

        return transcription.lower()
    except Exception as e:
        print(f"Error in speech processing: {e}")
        return None

# -------------------------------
# 6. Text Processing Function with LLaMA
# -------------------------------

def process_text_llama(input_text, llama_model):
    """
    Analyze input text using the LLaMA model via Ollama.
    """
    try:
        # Create a prompt
        prompt = f"Analyze the following patient statement for signs of medical emergency:\n\n\"{input_text}\"\n\nIs there an emergency? Provide a brief explanation."
        response = llama_model(prompt)
        if response:
            return response.strip()
        else:
            return "No response from LLaMA model."
    except Exception as e:
        print(f"Error in process_text_llama: {e}")
        return "Error analyzing text."

# -------------------------------
# 7. Data Fusion and Decision-Making
# -------------------------------

def data_fusion(physio_data, fall_detected, emotion, speech_text, speech_analysis, text_analysis):
    """
    Fuse data from various sources and make a decision using a medical model.
    """
    alerts = []
    warnings_list = []

    # Check physiological data
    if physio_data:
        if physio_data['heart_rate'] > 120 or physio_data['heart_rate'] < 50:
            alerts.append('Abnormal heart rate detected.')
        if physio_data['oxygen_saturation'] < 90.0:
            alerts.append('Low oxygen saturation detected.')
        if physio_data['blood_pressure_systolic'] > 140 or physio_data['blood_pressure_systolic'] < 90:
            alerts.append('Abnormal blood pressure detected.')
    else:
        warnings_list.append("Physiological data is missing or invalid.")

    # Check fall detection
    if fall_detected is True:
        alerts.append('Fall detected.')
    elif fall_detected is False:
        pass  # No fall detected
    else:
        warnings_list.append("Fall detection data is missing or invalid.")

    # Check emotion
    if emotion in ['sad', 'angry']:
        alerts.append(f'Negative emotion detected: {emotion}.')
    elif emotion is None:
        warnings_list.append("Emotion data is missing or invalid.")

    # Check speech content using LLaMA
    if speech_text and speech_analysis:
        if 'emergency' in speech_analysis.lower():
            alerts.append('Emergency detected in speech.')
    else:
        warnings_list.append("Speech analysis is missing or invalid.")

    # Check text input analysis
    if text_analysis:
        if 'emergency' in text_analysis.lower():
            alerts.append('Emergency detected in text input.')
    else:
        warnings_list.append("Text analysis is missing or invalid.")

    # Decision-making using a simple rule-based medical model
    if len(alerts) >= 2:
        decision = 'Medical emergency detected! Triggering alarm.'
    else:
        decision = 'No emergency detected.'

    return alerts, decision, warnings_list

# -------------------------------
# 8. Streamlit GUI Application
# -------------------------------

def medical_emergency_agent():
    st.title("Multimodal Medical Emergency Detection Agent")

    try:
        # Initialize LLaMA via Ollama
        llama_model = initialize_llama_via_ollama()

        # Generate synthetic data
        physio_data = generate_synthetic_data()
        st.subheader("Physiological Data")
        st.write(physio_data)

        # File uploads
        st.subheader("Upload Files for Analysis")

        # Video file upload
        video_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov'])
        if video_file is not None:
            video_path = os.path.join('temp_video', video_file.name)
            os.makedirs('temp_video', exist_ok=True)
            with open(video_path, 'wb') as f:
                f.write(video_file.getbuffer())
            fall_detected = process_video(video_path)
        else:
            st.info("No video file uploaded. Using default 'fall.mp4'.")
            video_path = 'fall.mp4'
            if not os.path.exists(video_path):
                st.error(f"Default video file '{video_path}' not found.")
                fall_detected = None
            else:
                fall_detected = process_video(video_path)

        # Image file upload
        image_file = st.file_uploader("Upload an image file", type=['jpg', 'jpeg', 'png'])
        if image_file is not None:
            image_path = os.path.join('temp_image', image_file.name)
            os.makedirs('temp_image', exist_ok=True)
            with open(image_path, 'wb') as f:
                f.write(image_file.getbuffer())
            emotion = process_image(image_path)
        else:
            st.info("No image file uploaded. Using default 'pain.jpg'.")
            image_path = 'pain.jpg'
            if not os.path.exists(image_path):
                st.error(f"Default image file '{image_path}' not found.")
                emotion = None
            else:
                emotion = process_image(image_path)

        # Audio file upload
        audio_file = st.file_uploader("Upload an audio file", type=['wav', 'mp3'])
        if audio_file is not None:
            audio_path = os.path.join('temp_audio', audio_file.name)
            os.makedirs('temp_audio', exist_ok=True)
            with open(audio_path, 'wb') as f:
                f.write(audio_file.getbuffer())
            speech_text = process_speech(audio_path)
        else:
            st.info("No audio file uploaded.")
            speech_text = None

        # JSON file upload
        json_file = st.file_uploader("Upload a JSON file", type=['json'])
        if json_file is not None:
            try:
                json_data = json.load(json_file)
                input_text = json_data.get('text', "No text found in JSON.")
            except Exception as e:
                st.error(f"Error reading uploaded JSON file: {e}")
                input_text = "Error reading JSON file."
        else:
            st.info("No JSON file uploaded. Using default 'database.json'.")
            if not os.path.exists('database.json'):
                st.error("Default JSON file 'database.json' not found.")
                input_text = "No input provided."
            else:
                try:
                    with open('database.json', 'r') as f:
                        json_data = json.load(f)
                        input_text = json_data.get('text', "No text found in JSON.")
                except Exception as e:
                    st.error(f"Error reading default JSON file: {e}")
                    input_text = "Error reading JSON file."

        # Display uploaded data
        st.subheader("Analysis Results")
        st.write(f"Fall Detected: {fall_detected}")
        st.write(f"Detected Emotion: {emotion}")
        st.write(f"Transcribed Speech: {speech_text}")
        st.write(f"Text from JSON: {input_text}")

        # Analyze speech text using LLaMA
        if speech_text:
            speech_analysis = process_text_llama(speech_text, llama_model)
        else:
            speech_analysis = None

        # Process text data using LLaMA
        text_analysis = process_text_llama(input_text, llama_model)

        # Display LLaMA analysis
        st.subheader("LLaMA Analysis")
        if speech_analysis:
            st.write(f"Speech Analysis:\n{speech_analysis}")
        else:
            st.write("No speech input or analysis.")

        if text_analysis:
            st.write(f"Text Analysis:\n{text_analysis}")
        else:
            st.write("No text analysis available.")

        # Perform data fusion and make decision
        alerts, decision, warnings_list = data_fusion(
            physio_data, fall_detected, emotion, speech_text, speech_analysis, text_analysis
        )

        # Display alerts and decision
        st.subheader("Alerts")
        if alerts:
            for alert in alerts:
                st.write(f"- {alert}")
        else:
            st.write("No alerts.")

        st.subheader("Decision")
        st.write(decision)

        if warnings_list:
            st.subheader("Warnings")
            for warning in warnings_list:
                st.warning(warning)

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# -------------------------------
# 9. Run the Streamlit App
# -------------------------------

if __name__ == "__main__":
    medical_emergency_agent()
