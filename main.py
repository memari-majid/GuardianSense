# Multimodal Medical Emergency Detection Agent with Q&A via Ollama LLaMA

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
                "model": "llama2",  # Updated model name
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
                print(f"Response: {response.text}")
                return ""
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with Ollama: {e}")
            return ""
        except Exception as e:
            print(f"Exception in llama_model: {e}")
            return ""
    return llama_model

# [Rest of the code remains the same, except for replacing 'pipelinel.png' with 'AI_doctor.png']

# -------------------------------
# 8. Streamlit GUI Application
# -------------------------------

def medical_emergency_agent():
    add_custom_css()

    # Display the AI doctor image
    st.image('AI_doctor.png', use_column_width=True)

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
        # [Upload Files code remains the same...]
        pass  # Code omitted for brevity

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
