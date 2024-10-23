# Multimodal Medical Emergency Detection Agent with Q&A via Ollama LLaMA 3.2 RAG System

# Import standard libraries
import random          # For generating random synthetic data and simulating processes
import time            # For time-related functions, if needed in the future
import cv2             # OpenCV library for video processing
import numpy as np      # For numerical operations, especially with arrays
import torch           # PyTorch library for deep learning models
import librosa         # For audio processing and feature extraction
import warnings        # To manage warning messages

# Suppress all warnings to keep the output clean
warnings.filterwarnings("ignore")

# Import necessary libraries for Ollama integration
import requests        # To make HTTP requests to the Ollama API
import json            # To handle JSON data
import os              # For interacting with the operating system (e.g., file handling)
import base64          # For encoding binary data to base64 (useful for embedding media)

# Import models for image and speech processing
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor  # Pre-trained models for speech-to-text
from PIL import Image                                     # For image processing

# Import Streamlit for creating the web-based GUI
import streamlit as st

# -------------------------------
# Custom CSS for UI Enhancement
# -------------------------------

def add_custom_css():
    """
    Adds custom CSS styles to the Streamlit app to enhance the UI appearance.
    """
    st.markdown(
        """
        <style>
        /* Set the default font for the body */
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        /* Set the background color for the main content area */
        .main {
            background-color: #f5f5f5;
        }
        /* Style for Streamlit buttons */
        .stButton > button {
            background-color: #4CAF50; /* Green background */
            color: white;              /* White text */
        }
        /* Style for the emergency alert banner */
        .emergency-alert {
            background-color: red;     /* Red background to indicate urgency */
            color: white;              /* White text for contrast */
            font-size: 24px;           /* Larger font size */
            text-align: center;        /* Centered text */
            padding: 20px;             /* Padding around the content */
            border-radius: 10px;       /* Rounded corners */
            animation: blink 1s infinite; /* Blinking animation to attract attention */
        }
        /* Keyframes for the blinking animation */
        @keyframes blink {
            0% {opacity: 1;}
            50% {opacity: 0.5;}
            100% {opacity: 1;}
        }
        </style>
        """,
        unsafe_allow_html=True  # Allow raw HTML for custom styling
    )

# -------------------------------
# Function to Encode Audio File
# -------------------------------

def get_audio_base64(file_path):
    """
    Reads an audio file from the given file path and encodes it to base64.
    This is useful for embedding audio directly into HTML.

    Parameters:
    - file_path (str): The path to the audio file to be encoded.

    Returns:
    - str or None: The base64-encoded string of the audio file, or None if an error occurs.
    """
    try:
        with open(file_path, 'rb') as f:
            data = f.read()  # Read the binary data of the audio file
        data_base64 = base64.b64encode(data).decode('utf-8')  # Encode to base64 and decode to string
        return data_base64
    except Exception as e:
        print(f"Error encoding audio file: {e}")  # Log the error
        return None  # Return None if encoding fails

# -------------------------------
# 1. Initialize LLaMA via Ollama
# -------------------------------

def initialize_llama_via_ollama():
    """
    Initializes the LLaMA model via the Ollama API.
    This function sets up a closure that can be used to send prompts to the LLaMA model
    and receive generated responses.

    Returns:
    - function: A function that takes a prompt string and returns the model's response.
    """
    base_url = "http://localhost:11434"  # Base URL for the Ollama API, typically running locally

    def llama_model(prompt):
        """
        Sends a prompt to the LLaMA model via Ollama and retrieves the generated response.

        Parameters:
        - prompt (str): The input text prompt to send to the model.

        Returns:
        - str: The generated response from the LLaMA model.
        """
        try:
            headers = {"Content-Type": "application/json"}  # Set the content type for the request
            data = {
                "model": "llama3.2",  # Specify the LLaMA 3.2 RAG model
                "prompt": prompt      # The prompt to send to the model
            }
            # Send a POST request to the Ollama API's generate endpoint with streaming enabled
            response = requests.post(
                f"{base_url}/api/generate",
                headers=headers,
                data=json.dumps(data),
                stream=True  # Enable streaming to receive the response incrementally
            )
            if response.status_code == 200:
                output = ""  # Initialize an empty string to accumulate the response
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')  # Decode the byte stream to string
                        try:
                            json_data = json.loads(decoded_line)  # Parse the JSON data
                            output += json_data.get('response', '')  # Append the response part
                        except json.JSONDecodeError:
                            continue  # If JSON is invalid, skip to the next line
                return output.strip()  # Return the accumulated response without leading/trailing whitespace
            else:
                # Log errors if the response status is not OK
                print(f"Error communicating with Ollama: {response.status_code}")
                print(f"Response: {response.text}")
                return ""  # Return an empty string on error
        except requests.exceptions.RequestException as e:
            # Handle exceptions related to the HTTP request
            print(f"Error communicating with Ollama: {e}")
            return ""
        except Exception as e:
            # Handle any other unexpected exceptions
            print(f"Exception in llama_model: {e}")
            return ""

    return llama_model  # Return the closure function

# -------------------------------
# 2. Synthetic Data Generation
# -------------------------------

def generate_synthetic_data():
    """
    Generates synthetic physiological data to mimic data that might be collected from a device like an Apple Watch.
    This data includes heart rate, oxygen saturation, blood pressure, steps, calories burned, and sleep hours.

    Returns:
    - dict: A dictionary containing the synthetic physiological data.
    """
    data = {
        'heart_rate': random.randint(50, 150),                # Heart rate in beats per minute (bpm)
        'oxygen_saturation': random.uniform(85, 100),         # Oxygen saturation in percentage (%)
        'blood_pressure_systolic': random.randint(90, 160),    # Systolic blood pressure in mmHg
        'blood_pressure_diastolic': random.randint(60, 100),   # Diastolic blood pressure in mmHg
        'steps': random.randint(0, 10000),                     # Number of steps taken
        'calories_burned': random.uniform(0, 500),             # Calories burned in kilocalories (kcal)
        'sleep_hours': random.uniform(0, 12),                  # Sleep duration in hours
    }
    return data  # Return the generated data

# -------------------------------
# 3. Video Processing Function
# -------------------------------

def process_video(video_path):
    """
    Processes a video file to extract features relevant to medical emergency detection, such as fall detection.
    Currently, this function simulates video processing by randomly determining if a fall is detected.

    Parameters:
    - video_path (str): The file path to the video to be processed.

    Returns:
    - bool: True if a fall is detected, False otherwise.
    """
    try:
        # Check if the video file exists
        if not os.path.exists(video_path):
            print(f"Error: Video file {video_path} does not exist.")
            return False  # Return False if the video file is missing

        # For simplicity, simulate video processing with random fall detection
        fall_detected = random.choice([True, False])
        return fall_detected  # Return the simulated result
    except Exception as e:
        # Handle any exceptions during video processing
        print(f"Error in video processing: {e}")
        return False  # Default to False on error

# -------------------------------
# 4. Image Processing Function
# -------------------------------

def process_image(image_path):
    """
    Analyzes an image to detect facial expressions, simulating emotion detection.

    Parameters:
    - image_path (str): The file path to the image to be processed.

    Returns:
    - str or None: The detected dominant emotion or None if an error occurs.
    """
    try:
        # Check if the image file exists
        if not os.path.exists(image_path):
            print(f"Error: Image file {image_path} does not exist.")
            return None  # Return None if the image file is missing

        # Load the image to ensure it's readable
        img = Image.open(image_path)

        # Simulate emotion detection by randomly selecting an emotion
        emotions = ['happy', 'sad', 'angry', 'surprised', 'neutral']
        dominant_emotion = random.choice(emotions)
        return dominant_emotion  # Return the simulated emotion
    except Exception as e:
        # Handle any exceptions during image processing
        print(f"Error in image processing: {e}")
        return None  # Return None on error

# -------------------------------
# 5. Speech Processing Function
# -------------------------------

def process_speech(audio_path):
    """
    Converts speech in an audio file to text using a pre-trained Wav2Vec2 model.

    Parameters:
    - audio_path (str): The file path to the audio file to be processed.

    Returns:
    - str or None: The transcribed text in lowercase or None if an error occurs.
    """
    try:
        # Check if the audio file exists
        if not os.path.exists(audio_path):
            print(f"Error: Audio file {audio_path} does not exist.")
            return None  # Return None if the audio file is missing

        # Load pre-trained Wav2Vec2 processor and model for speech-to-text
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

        # Load the audio file using librosa, resampling to 16kHz
        speech, rate = librosa.load(audio_path, sr=16000)

        # Check if the audio file is empty or unreadable
        if len(speech) == 0:
            print("Error: Audio file is empty or cannot be read.")
            return None  # Return None if audio data is invalid

        # Tokenize the audio input for the model
        input_values = processor(speech, return_tensors='pt', padding='longest').input_values

        # Perform inference to get logits from the model
        logits = model(input_values).logits

        # Get the predicted token IDs by taking the argmax of logits
        predicted_ids = torch.argmax(logits, dim=-1)

        # Decode the predicted token IDs to get the transcribed text
        transcription = processor.decode(predicted_ids[0])

        return transcription.lower()  # Return the transcription in lowercase
    except Exception as e:
        # Handle any exceptions during speech processing
        print(f"Error in speech processing: {e}")
        return None  # Return None on error

# -------------------------------
# 6. Text Processing Function with LLaMA
# -------------------------------

def process_text_llama(input_text, llama_model):
    """
    Analyzes input text using the LLaMA model via Ollama to determine signs of medical emergency.

    Parameters:
    - input_text (str): The text input to be analyzed.
    - llama_model (function): The LLaMA model function initialized via Ollama.

    Returns:
    - str: The analysis result from the LLaMA model.
    """
    try:
        # Create a prompt that instructs the model to analyze the patient statement
        prompt = f"Analyze the following patient statement for signs of medical emergency:\n\n\"{input_text}\"\n\nIs there an emergency? Provide a brief explanation."
        
        # Get the response from the LLaMA model
        response = llama_model(prompt)
        
        if response:
            return response.strip()  # Return the trimmed response if available
        else:
            return "No response from LLaMA model."  # Default message if no response
    except Exception as e:
        # Handle any exceptions during text processing
        print(f"Error in process_text_llama: {e}")
        return "Error analyzing text."  # Return an error message

# -------------------------------
# 7. Data Fusion and Decision-Making
# -------------------------------

def data_fusion(physio_data, fall_detected, emotion, speech_text, speech_analysis, text_analysis):
    """
    Fuses data from various sources (physiological, video, image, audio, text) to make a decision
    about whether a medical emergency has occurred.

    Parameters:
    - physio_data (dict): Physiological data metrics.
    - fall_detected (bool): Whether a fall was detected in the video.
    - emotion (str): Detected emotion from the image.
    - speech_text (str): Transcribed speech text.
    - speech_analysis (str): Analysis of the speech text by LLaMA.
    - text_analysis (str): Analysis of the JSON text input by LLaMA.

    Returns:
    - tuple: A tuple containing alerts (list), decision message (str), warnings (list), and emergency_detected (bool).
    """
    alerts = []          # List to store alert messages
    warnings_list = []   # List to store warning messages

    # -------------------------------
    # Check Physiological Data
    # -------------------------------
    if physio_data:
        # Check for abnormal heart rate
        if physio_data['heart_rate'] > 120 or physio_data['heart_rate'] < 50:
            alerts.append('Abnormal heart rate detected.')
        # Check for low oxygen saturation
        if physio_data['oxygen_saturation'] < 90.0:
            alerts.append('Low oxygen saturation detected.')
        # Check for abnormal blood pressure
        if physio_data['blood_pressure_systolic'] > 140 or physio_data['blood_pressure_systolic'] < 90:
            alerts.append('Abnormal blood pressure detected.')
    else:
        # Warn if physiological data is missing or invalid
        warnings_list.append("Physiological data is missing or invalid.")

    # -------------------------------
    # Check Fall Detection
    # -------------------------------
    if fall_detected is True:
        alerts.append('Fall detected.')
    elif fall_detected is False:
        pass  # No fall detected; no action needed
    else:
        # Warn if fall detection data is missing or invalid
        warnings_list.append("Fall detection data is missing or invalid.")

    # -------------------------------
    # Check Emotion Detection
    # -------------------------------
    if emotion in ['sad', 'angry']:
        alerts.append(f'Negative emotion detected: {emotion}.')
    elif emotion is None:
        # Warn if emotion data is missing or invalid
        warnings_list.append("Emotion data is missing or invalid.")

    # -------------------------------
    # Check Speech Content Using LLaMA
    # -------------------------------
    if speech_text and speech_analysis:
        # If the word 'emergency' is detected in the speech analysis, trigger an alert
        if 'emergency' in speech_analysis.lower():
            alerts.append('Emergency detected in speech.')
    else:
        # Warn if speech analysis is missing or invalid
        warnings_list.append("Speech analysis is missing or invalid.")

    # -------------------------------
    # Check Text Input Analysis
    # -------------------------------
    if text_analysis:
        # If the word 'emergency' is detected in the text analysis, trigger an alert
        if 'emergency' in text_analysis.lower():
            alerts.append('Emergency detected in text input.')
    else:
        # Warn if text analysis is missing or invalid
        warnings_list.append("Text analysis is missing or invalid.")

    # -------------------------------
    # Decision-Making Using a Simple Rule-Based Model
    # -------------------------------
    if len(alerts) >= 2:
        # If there are two or more alerts, consider it a medical emergency
        decision = 'Medical emergency detected! Triggering alarm.'
        emergency_detected = True
    else:
        # Otherwise, no emergency is detected
        decision = 'No emergency detected.'
        emergency_detected = False

    return alerts, decision, warnings_list, emergency_detected  # Return all relevant information

# -------------------------------
# 8. Streamlit GUI Application
# -------------------------------

def medical_emergency_agent():
    """
    The main function that runs the Streamlit web application for the Medical Emergency Detection Agent.
    It handles the UI, file uploads, data processing, analysis, and displays results to the user.
    """
    add_custom_css()  # Apply custom CSS styles to enhance the UI

    # Display the AI doctor image at the top of the app
    st.image('AI_doctor.png', use_column_width=True)

    # Set the main title of the application
    st.title("Multimodal Medical Emergency Detection Agent")

    # -------------------------------
    # Sidebar for Navigation and Exit Button
    # -------------------------------
    st.sidebar.title("Navigation")  # Title for the sidebar
    # Dropdown menu for selecting the app mode
    app_mode = st.sidebar.selectbox("Choose the app mode",
                                    ["Home", "Upload Files", "View Results"])

    # Exit Button in Sidebar
    if st.sidebar.button("Exit Application"):
        st.sidebar.write("Exiting the application...")  # Inform the user
        st.stop()  # Stop the Streamlit app

    # -------------------------------
    # Home Page
    # -------------------------------
    if app_mode == "Home":
        st.header("Welcome!")  # Header for the Home page
        st.write("""
            This application detects medical emergencies by analyzing multimodal inputs, including physiological data, video, images, audio, and text. 
            Please navigate to **Upload Files** to provide input data, or proceed to **View Results** to see the analysis.
        """)  # Description of the app's functionality

    # -------------------------------
    # Upload Files Page
    # -------------------------------
    elif app_mode == "Upload Files":
        st.header("Upload Files for Analysis")  # Header for the Upload Files page

        try:
            # Initialize LLaMA via Ollama
            llama_model = initialize_llama_via_ollama()

            # Generate synthetic physiological data
            physio_data = generate_synthetic_data()
            st.subheader("Physiological Data")  # Subheader for physiological data
            st.write(physio_data)  # Display the synthetic physiological data

            # -------------------------------
            # File Uploads Section
            # -------------------------------
            st.subheader("File Uploads")  # Subheader for file uploads

            # -------------------------------
            # Video File Upload
            # -------------------------------
            video_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov'])  # File uploader for video
            if video_file is not None:
                # If a video file is uploaded, save it to a temporary directory
                video_path = os.path.join('temp_video', video_file.name)
                os.makedirs('temp_video', exist_ok=True)  # Create the directory if it doesn't exist
                with open(video_path, 'wb') as f:
                    f.write(video_file.getbuffer())  # Write the uploaded file's bytes to disk
                fall_detected = process_video(video_path)  # Process the uploaded video
            else:
                # If no video is uploaded, use a default video file
                st.info("No video file uploaded. Using default 'fall.mp4'.")
                video_path = 'fall.mp4'
                if not os.path.exists(video_path):
                    st.error(f"Default video file '{video_path}' not found.")  # Error if default video is missing
                    fall_detected = None
                else:
                    fall_detected = process_video(video_path)  # Process the default video

            # -------------------------------
            # Image File Upload
            # -------------------------------
            image_file = st.file_uploader("Upload an image file", type=['jpg', 'jpeg', 'png'])  # File uploader for image
            if image_file is not None:
                # If an image file is uploaded, save it to a temporary directory
                image_path = os.path.join('temp_image', image_file.name)
                os.makedirs('temp_image', exist_ok=True)  # Create the directory if it doesn't exist
                with open(image_path, 'wb') as f:
                    f.write(image_file.getbuffer())  # Write the uploaded file's bytes to disk
                emotion = process_image(image_path)  # Process the uploaded image
            else:
                # If no image is uploaded, use a default image file
                st.info("No image file uploaded. Using default 'pain.jpg'.")
                image_path = 'pain.jpg'
                if not os.path.exists(image_path):
                    st.error(f"Default image file '{image_path}' not found.")  # Error if default image is missing
                    emotion = None
                else:
                    emotion = process_image(image_path)  # Process the default image

            # -------------------------------
            # Audio File Upload
            # -------------------------------
            audio_file = st.file_uploader("Upload an audio file", type=['wav', 'mp3'])  # File uploader for audio
            if audio_file is not None:
                # If an audio file is uploaded, save it to a temporary directory
                audio_path = os.path.join('temp_audio', audio_file.name)
                os.makedirs('temp_audio', exist_ok=True)  # Create the directory if it doesn't exist
                with open(audio_path, 'wb') as f:
                    f.write(audio_file.getbuffer())  # Write the uploaded file's bytes to disk
                speech_text = process_speech(audio_path)  # Process the uploaded audio
            else:
                # If no audio is uploaded, set speech_text to None
                st.info("No audio file uploaded.")
                speech_text = None

            # -------------------------------
            # JSON File Upload
            # -------------------------------
            json_file = st.file_uploader("Upload a JSON file", type=['json'])  # File uploader for JSON
            if json_file is not None:
                try:
                    # Attempt to load the uploaded JSON file
                    json_data = json.load(json_file)
                    input_text = json_data.get('text', "No text found in JSON.")  # Extract 'text' field
                except Exception as e:
                    st.error(f"Error reading uploaded JSON file: {e}")  # Error if JSON is invalid
                    input_text = "Error reading JSON file."
            else:
                # If no JSON is uploaded, use a default JSON file
                st.info("No JSON file uploaded. Using default 'database.json'.")
                if not os.path.exists('database.json'):
                    st.error("Default JSON file 'database.json' not found.")  # Error if default JSON is missing
                    input_text = "No input provided."
                else:
                    try:
                        with open('database.json', 'r') as f:
                            json_data = json.load(f)  # Load the default JSON file
                            input_text = json_data.get('text', "No text found in JSON.")  # Extract 'text' field
                    except Exception as e:
                        st.error(f"Error reading default JSON file: {e}")  # Error if default JSON is invalid
                        input_text = "Error reading JSON file."

            # -------------------------------
            # Save Session State
            # -------------------------------
            # Store all relevant data in Streamlit's session state for later use
            st.session_state['physio_data'] = physio_data
            st.session_state['fall_detected'] = fall_detected
            st.session_state['emotion'] = emotion
            st.session_state['speech_text'] = speech_text
            st.session_state['input_text'] = input_text
            st.session_state['llama_model'] = llama_model

            # Inform the user that files have been uploaded and processed successfully
            st.success("Files uploaded and processed successfully! Navigate to 'View Results' to see the analysis.")

        except Exception as e:
            # Handle any unexpected errors during the upload and processing phase
            st.error(f"An unexpected error occurred: {e}")

    # -------------------------------
    # View Results Page
    # -------------------------------
    elif app_mode == "View Results":
        st.header("Analysis Results")  # Header for the View Results page

        # Check if data has been uploaded; if not, prompt the user to upload files first
        if 'physio_data' not in st.session_state:
            st.warning("No data found. Please upload files first.")
            return  # Exit the function if no data is present

        # Retrieve all necessary data from the session state
        physio_data = st.session_state.get('physio_data', {})
        fall_detected = st.session_state.get('fall_detected', None)
        emotion = st.session_state.get('emotion', None)
        speech_text = st.session_state.get('speech_text', None)
        input_text = st.session_state.get('input_text', "")
        llama_model = st.session_state.get('llama_model', None)

        # -------------------------------
        # Display Uploaded and Processed Data
        # -------------------------------
        st.subheader("Physiological Data")  # Subheader for physiological data
        st.write(physio_data)  # Display the physiological data

        st.subheader("Uploaded Data Analysis")  # Subheader for uploaded data analysis
        st.write(f"**Fall Detected:** {fall_detected}")           # Display fall detection result
        st.write(f"**Detected Emotion:** {emotion}")             # Display detected emotion
        st.write(f"**Transcribed Speech:** {speech_text}")       # Display transcribed speech text
        st.write(f"**Text from JSON:** {input_text}")            # Display text from JSON input

        # -------------------------------
        # Analyze Speech Text Using LLaMA
        # -------------------------------
        with st.spinner('Analyzing speech text with AI Doctor...'):  # Show a spinner during processing
            if speech_text and llama_model:
                # Analyze the transcribed speech text using LLaMA
                speech_analysis = process_text_llama(speech_text, llama_model)
            else:
                speech_analysis = None  # Set to None if no speech text or model is available

        # -------------------------------
        # Analyze Text Input Using LLaMA
        # -------------------------------
        with st.spinner('Analyzing text input with AI Doctor...'):  # Show a spinner during processing
            if input_text and llama_model:
                # Analyze the JSON text input using LLaMA
                text_analysis = process_text_llama(input_text, llama_model)
            else:
                text_analysis = None  # Set to None if no input text or model is available

        # -------------------------------
        # Display LLaMA Analysis
        # -------------------------------
        st.subheader("Analysis by AI Doctor")  # Subheader for AI Doctor's analysis
        st.info("The data has been analyzed by the AI Doctor.")  # Inform the user

        st.subheader("LLaMA Analysis")  # Subheader for LLaMA-specific analysis
        if speech_analysis:
            st.write(f"**Speech Analysis:**\n{speech_analysis}")  # Display speech analysis result
        else:
            st.write("No speech input or analysis.")  # Inform if speech analysis is unavailable

        if text_analysis:
            st.write(f"**Text Analysis:**\n{text_analysis}")  # Display text analysis result
        else:
            st.write("No text analysis available.")  # Inform if text analysis is unavailable

        # -------------------------------
        # Perform Data Fusion and Make Decision
        # -------------------------------
        alerts, decision, warnings_list, emergency_detected = data_fusion(
            physio_data, fall_detected, emotion, speech_text, speech_analysis, text_analysis
        )

        # -------------------------------
        # Display Alerts and Decision
        # -------------------------------
        st.subheader("Alerts")  # Subheader for alerts
        if alerts:
            for alert in alerts:
                st.error(f"- {alert}")  # Display each alert as an error message
        else:
            st.write("No alerts.")  # Inform if there are no alerts

        st.subheader("Decision")  # Subheader for the decision message
        if emergency_detected:
            # If an emergency is detected, display a graphical alert and play an alarm sound

            # Display a styled emergency alert banner with an alarm icon
            st.markdown(
                """
                <div class="emergency-alert">
                    <img src="https://img.icons8.com/emoji/48/alarm-clock-emoji.png" width="50" style="vertical-align: middle;"/>
                    <span style="vertical-align: middle;">Medical Emergency Detected! Triggering Alarm!</span>
                </div>
                """,
                unsafe_allow_html=True  # Allow raw HTML for styling
            )
            st.warning("Emergency detected! Please respond immediately.")  # Warning message to the user

            # Play alarm sound automatically by embedding the audio
            audio_base64 = get_audio_base64('alarm_sound.mp3')  # Encode the alarm sound to base64
            if audio_base64:
                st.markdown(
                    f"""
                    <audio autoplay>
                        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                    </audio>
                    """,
                    unsafe_allow_html=True  # Embed the audio player with autoplay
                )
            else:
                st.error("Alarm sound file not found or could not be read.")  # Error if audio file is missing

        else:
            st.success(decision)  # Display the decision message as a success message

        # -------------------------------
        # Display Warnings, If Any
        # -------------------------------
        if warnings_list:
            st.subheader("Warnings")  # Subheader for warnings
            for warning in warnings_list:
                st.warning(warning)  # Display each warning as a warning message

        # -------------------------------
        # Note to User About Autoplay Policies
        # -------------------------------
        if emergency_detected:
            st.info("Note: If you do not hear the alarm sound, your browser may have blocked autoplay. Please adjust your browser settings to allow autoplay of audio.")

        # -------------------------------
        # New Section: Ask Questions with Fact-Checking
        # -------------------------------

        st.header("Ask Questions about the Data")  # Header for the Q&A section

        if llama_model:
            # Prepare a summary of all data to provide context for the AI Doctor
            data_summary = f"""
            Physiological Data:
            {physio_data}

            Fall Detected: {fall_detected}
            Detected Emotion: {emotion}
            Transcribed Speech: {speech_text}
            Text from JSON: {input_text}
            Alerts: {alerts}
            """

            st.write("You can ask questions about the uploaded data. The AI Doctor will answer your questions based on the data provided.")  # Instructions for the user

            # Text input for the user to enter a question
            question = st.text_input("Enter your question:")
            if question:
                with st.spinner('Generating answer...'):  # Show a spinner while generating the answer
                    # Create a prompt that includes the data summary and the user's question
                    prompt = f"""
                    Based on the following data:

                    {data_summary}

                    Answer the following question:

                    {question}
                    """
                    answer = llama_model(prompt)  # Get the answer from the LLaMA model
                    if answer:
                        st.subheader("Answer:")  # Subheader for the answer
                        st.write(answer)  # Display the answer

                        # -------------------------------
                        # Fact-Check the Answer Using the Medical Model
                        # -------------------------------
                        with st.spinner('Fact-checking the answer with the medical model...'):  # Show a spinner during fact-checking
                            # Create a prompt to fact-check the provided answer
                            fact_check_prompt = f"""
                            Based on the following data:

                            {data_summary}

                            The following answer was provided to the question "{question}":

                            {answer}

                            Is this answer correct based on the data provided? Provide a brief explanation and correct any inaccuracies.
                            """
                            fact_check_result = llama_model(fact_check_prompt)  # Get the fact-check result from the model
                            if fact_check_result:
                                st.subheader("Fact-Check Result:")  # Subheader for the fact-check result
                                st.write(fact_check_result)  # Display the fact-check result
                            else:
                                st.error("No response from the medical model during fact-checking.")  # Error if fact-checking fails
                    else:
                        st.error("No response from the AI Doctor.")  # Error if no answer is generated
        else:
            st.error("LLaMA model is not initialized.")  # Error if the LLaMA model is unavailable

# -------------------------------
# 9. Run the Streamlit App
# -------------------------------

if __name__ == "__main__":
    medical_emergency_agent()  # Execute the main function to run the Streamlit app
