# Multimodal Medical Emergency Detection Agent with Q&A via Ollama LLaMA 3.2 RAG System
# it does not do Q&A via Ollama LLaMA 3.2 RAG System and AI doctor yet

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
import base64

# Import models for image and speech processing
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from PIL import Image

# Import Streamlit for web GUI
import streamlit as st

# -------------------------------
# Custom CSS for UI Enhancement
# -------------------------------

def add_custom_css():
    st.markdown(
        """
        <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .main {
            background-color: #f5f5f5;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
        }
        .emergency-alert {
            background-color: red;
            color: white;
            font-size: 24px;
            text-align: center;
            padding: 20px;
            border-radius: 10px;
            animation: blink 1s infinite;
        }
        @keyframes blink {
            0% {opacity: 1;}
            50% {opacity: 0.5;}
            100% {opacity: 1;}
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# -------------------------------
# Function to Encode Audio File
# -------------------------------

def get_audio_base64(file_path):
    """
    Reads an audio file and encodes it to base64 for embedding in HTML.
    """
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        data_base64 = base64.b64encode(data).decode('utf-8')
        return data_base64
    except Exception as e:
        print(f"Error encoding audio file: {e}")
        return None

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
                "model": "llama-3.2-rag",  # Specify LLaMA 3.2 RAG model
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
        emergency_detected = True
    else:
        decision = 'No emergency detected.'
        emergency_detected = False

    return alerts, decision, warnings_list, emergency_detected

# -------------------------------
# 8. Streamlit GUI Application
# -------------------------------

def medical_emergency_agent():
    add_custom_css()

    # Display the AI Doctor image
    st.image('AI_doctor.png', width=300)

    st.title("Multimodal Medical Emergency Detection Agent")

    # Sidebar for navigation and Exit button
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the app mode",
                                    ["Home", "Upload Files", "View Results"])

    # Exit Button in Sidebar
    if st.sidebar.button("Exit Application"):
        st.sidebar.write("Exiting the application...")
        st.stop()

    if app_mode == "Home":
        st.header("Welcome!")
        st.write("""
            This application detects medical emergencies by analyzing multimodal inputs, including physiological data, video, images, audio, and text. 
            Please navigate to **Upload Files** to provide input data, or proceed to **View Results** to see the analysis.
        """)
    elif app_mode == "Upload Files":
        st.header("Upload Files for Analysis")

        try:
            # Initialize LLaMA via Ollama
            llama_model = initialize_llama_via_ollama()

            # Generate synthetic data
            physio_data = generate_synthetic_data()
            st.subheader("Physiological Data")
            st.write(physio_data)

            # File uploads
            st.subheader("File Uploads")

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

            # Save session state
            st.session_state['physio_data'] = physio_data
            st.session_state['fall_detected'] = fall_detected
            st.session_state['emotion'] = emotion
            st.session_state['speech_text'] = speech_text
            st.session_state['input_text'] = input_text
            st.session_state['llama_model'] = llama_model

            st.success("Files uploaded and processed successfully! Navigate to 'View Results' to see the analysis.")

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

    elif app_mode == "View Results":
        st.header("Analysis Results")

        if 'physio_data' not in st.session_state:
            st.warning("No data found. Please upload files first.")
            return

        physio_data = st.session_state.get('physio_data', {})
        fall_detected = st.session_state.get('fall_detected', None)
        emotion = st.session_state.get('emotion', None)
        speech_text = st.session_state.get('speech_text', None)
        input_text = st.session_state.get('input_text', "")
        llama_model = st.session_state.get('llama_model', None)

        st.subheader("Physiological Data")
        st.write(physio_data)

        st.subheader("Uploaded Data Analysis")
        st.write(f"**Fall Detected:** {fall_detected}")
        st.write(f"**Detected Emotion:** {emotion}")
        st.write(f"**Transcribed Speech:** {speech_text}")
        st.write(f"**Text from JSON:** {input_text}")

        # Analyze speech text using LLaMA
        with st.spinner('Analyzing speech text with AI Doctor...'):
            if speech_text and llama_model:
                speech_analysis = process_text_llama(speech_text, llama_model)
            else:
                speech_analysis = None

        # Process text data using LLaMA
        with st.spinner('Analyzing text input with AI Doctor...'):
            if input_text and llama_model:
                text_analysis = process_text_llama(input_text, llama_model)
            else:
                text_analysis = None

        # Display LLaMA analysis
        st.subheader("Analysis by AI Doctor")
        st.info("The data has been analyzed by the AI Doctor.")

        st.subheader("LLaMA Analysis")
        if speech_analysis:
            st.write(f"**Speech Analysis:**\n{speech_analysis}")
        else:
            st.write("No speech input or analysis.")

        if text_analysis:
            st.write(f"**Text Analysis:**\n{text_analysis}")
        else:
            st.write("No text analysis available.")

        # Perform data fusion and make decision
        alerts, decision, warnings_list, emergency_detected = data_fusion(
            physio_data, fall_detected, emotion, speech_text, speech_analysis, text_analysis
        )

        # Display alerts and decision
        st.subheader("Alerts")
        if alerts:
            for alert in alerts:
                st.error(f"- {alert}")
        else:
            st.write("No alerts.")

        st.subheader("Decision")
        if emergency_detected:
            # Display graphical emergency alert
            st.markdown(
                """
                <div class="emergency-alert">
                    <img src="https://img.icons8.com/emoji/48/alarm-clock-emoji.png" width="50" style="vertical-align: middle;"/>
                    <span style="vertical-align: middle;">Medical Emergency Detected! Triggering Alarm!</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.warning("Emergency detected! Please respond immediately.")

            # Play alarm sound automatically
            audio_base64 = get_audio_base64('alarm_sound.mp3')
            if audio_base64:
                st.markdown(
                    f"""
                    <audio autoplay>
                        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                    </audio>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.error("Alarm sound file not found or could not be read.")
        else:
            st.success(decision)

        if warnings_list:
            st.subheader("Warnings")
            for warning in warnings_list:
                st.warning(warning)

        # Note to user about autoplay policies
        if emergency_detected:
            st.info("Note: If you do not hear the alarm sound, your browser may have blocked autoplay. Please adjust your browser settings to allow autoplay of audio.")

        # -------------------------------
        # New Section: Ask Questions
        # -------------------------------

        st.header("Ask Questions about the Data")

        if llama_model:
            # Prepare data summary as context
            data_summary = f"""
            Physiological Data:
            {physio_data}

            Fall Detected: {fall_detected}
            Detected Emotion: {emotion}
            Transcribed Speech: {speech_text}
            Text from JSON: {input_text}
            Alerts: {alerts}
            """

            st.write("You can ask questions about the uploaded data. The AI Doctor will answer your questions based on the data provided.")

            question = st.text_input("Enter your question:")
            if question:
                with st.spinner('Generating answer...'):
                    # Create prompt for LLaMA model
                    prompt = f"""
                    Based on the following data:

                    {data_summary}

                    Answer the following question:

                    {question}
                    """
                    answer = llama_model(prompt)
                    if answer:
                        st.subheader("Answer:")
                        st.write(answer)
                    else:
                        st.error("No response from the AI Doctor.")
        else:
            st.error("LLaMA model is not initialized.")

# -------------------------------
# 9. Run the Streamlit App
# -------------------------------

if __name__ == "__main__":
    medical_emergency_agent()
