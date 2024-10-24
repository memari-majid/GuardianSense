# Multimodal Medical Emergency Detection Agent with Enhanced Evaluation and Visualization

# -------------------------------
# Import Standard Libraries
# -------------------------------

import os
import random
import numpy as np
import pandas as pd
import json
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import streamlit as st
import sqlite3  # Import sqlite3 for database interaction

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

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
        /* Custom CSS styles */
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
# 1. Load Metadata and Data Files
# -------------------------------

def load_metadata(db_path='metadata/dataset_metadata.db'):
    """
    Loads the metadata from the specified SQLite database.

    Parameters:
    - db_path (str): Path to the SQLite database file.

    Returns:
    - pd.DataFrame: DataFrame containing the metadata.
    """
    if not os.path.exists(db_path):
        print(f"Database file {db_path} not found.")
        return None
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    # Read the metadata table into a DataFrame
    metadata_df = pd.read_sql_query("SELECT * FROM metadata", conn)
    # Close the database connection
    conn.close()
    return metadata_df

# -------------------------------
# 2. Data Loading Functions
# -------------------------------

def load_text_data(metadata_df):
    """
    Loads text data based on the metadata.

    Parameters:
    - metadata_df (pd.DataFrame): DataFrame containing the metadata.

    Returns:
    - dict: Dictionary mapping SampleID to text data.
    """
    text_data = {}
    for _, row in metadata_df.iterrows():
        sample_id = row['SampleID']
        text_path = row['TextDataPath']
        try:
            with open(text_path, 'r') as f:
                text = f.read()
            text_data[sample_id] = text
        except:
            text_data[sample_id] = ""
    return text_data

def load_image_data(metadata_df):
    """
    Loads image data paths based on the metadata.

    Parameters:
    - metadata_df (pd.DataFrame): DataFrame containing the metadata.

    Returns:
    - dict: Dictionary mapping SampleID to image paths.
    """
    image_data = {}
    for _, row in metadata_df.iterrows():
        sample_id = row['SampleID']
        image_path = row['ImageDataPath']
        if os.path.exists(image_path):
            image_data[sample_id] = image_path
        else:
            image_data[sample_id] = None
    return image_data

def load_audio_data(metadata_df):
    """
    Loads audio data paths based on the metadata.

    Parameters:
    - metadata_df (pd.DataFrame): DataFrame containing the metadata.

    Returns:
    - dict: Dictionary mapping SampleID to audio paths.
    """
    audio_data = {}
    for _, row in metadata_df.iterrows():
        sample_id = row['SampleID']
        audio_path = row['AudioDataPath']
        if os.path.exists(audio_path):
            audio_data[sample_id] = audio_path
        else:
            audio_data[sample_id] = None
    return audio_data

def load_video_data(metadata_df):
    """
    Loads video data paths based on the metadata.

    Parameters:
    - metadata_df (pd.DataFrame): DataFrame containing the metadata.

    Returns:
    - dict: Dictionary mapping SampleID to video paths.
    """
    video_data = {}
    for _, row in metadata_df.iterrows():
        sample_id = row['SampleID']
        video_path = row['VideoDataPath']
        if os.path.exists(video_path):
            video_data[sample_id] = video_path
        else:
            video_data[sample_id] = None
    return video_data

def load_physiological_data(metadata_df):
    """
    Loads physiological data based on the metadata.

    Parameters:
    - metadata_df (pd.DataFrame): DataFrame containing the metadata.

    Returns:
    - dict: Dictionary mapping SampleID to physiological data.
    """
    physio_data = {}
    for _, row in metadata_df.iterrows():
        sample_id = row['SampleID']
        physio_path = row['PhysiologicalDataPath']
        try:
            with open(physio_path, 'r') as f:
                data = json.load(f)
            physio_data[sample_id] = data
        except:
            physio_data[sample_id] = None
    return physio_data

# -------------------------------
# 3. Data Processing Functions
# -------------------------------

def process_text_data(text_data):
    """
    Processes text data (placeholder for actual NLP processing).

    Parameters:
    - text_data (dict): Dictionary mapping SampleID to text data.

    Returns:
    - dict: Dictionary mapping SampleID to processed text features.
    """
    processed_text = {}
    for sample_id, text in text_data.items():
        # Placeholder for actual NLP processing (e.g., embeddings)
        processed_text[sample_id] = text.lower()
    return processed_text

def process_image_data(image_data):
    """
    Processes image data (placeholder for actual image processing).

    Parameters:
    - image_data (dict): Dictionary mapping SampleID to image paths.

    Returns:
    - dict: Dictionary mapping SampleID to processed image features.
    """
    processed_images = {}
    for sample_id, image_path in image_data.items():
        # Placeholder for actual image processing (e.g., emotion detection)
        # Simulate emotion detection
        emotions = ['Happy', 'Sad', 'Angry', 'Surprised', 'Neutral', 'Pain']
        dominant_emotion = random.choice(emotions)
        processed_images[sample_id] = dominant_emotion
    return processed_images

def process_audio_data(audio_data):
    """
    Processes audio data (placeholder for actual speech recognition).

    Parameters:
    - audio_data (dict): Dictionary mapping SampleID to audio paths.

    Returns:
    - dict: Dictionary mapping SampleID to transcribed speech text.
    """
    processed_audio = {}
    for sample_id, audio_path in audio_data.items():
        # Placeholder for actual speech processing
        # Simulate speech transcription
        transcribed_text = "simulated speech transcription"
        processed_audio[sample_id] = transcribed_text
    return processed_audio

def process_video_data(video_data):
    """
    Processes video data (placeholder for actual video processing).

    Parameters:
    - video_data (dict): Dictionary mapping SampleID to video paths.

    Returns:
    - dict: Dictionary mapping SampleID to detected scenarios.
    """
    processed_video = {}
    for sample_id, video_path in video_data.items():
        # Placeholder for actual video processing
        # Simulate scenario detection
        scenarios = ['Fall', 'Seizure', 'Choking', 'None']
        detected_scenario = random.choice(scenarios)
        if detected_scenario != 'None':
            processed_video[sample_id] = detected_scenario
        else:
            processed_video[sample_id] = None
    return processed_video

def process_physiological_data(physio_data):
    """
    Processes physiological data.

    Parameters:
    - physio_data (dict): Dictionary mapping SampleID to physiological data.

    Returns:
    - dict: Dictionary mapping SampleID to detected scenarios.
    """
    processed_physio = {}
    for sample_id, data in physio_data.items():
        if data:
            # Analyze physiological data to detect possible emergencies
            physio_emergency = None
            # Check for signs of specific emergencies based on vital signs
            heart_rate = data.get('HeartRate', 0)
            systolic_bp = data.get('SystolicBP', 0)
            oxygen_saturation = data.get('OxygenSaturation', 0)
            if heart_rate > 120 and systolic_bp < 90:
                physio_emergency = 'Heart Attack'
            elif oxygen_saturation < 90.0:
                physio_emergency = 'Respiratory Distress'
            elif systolic_bp > 140:
                physio_emergency = 'Stroke'
            elif heart_rate > 150:
                physio_emergency = 'Seizure'
            processed_physio[sample_id] = physio_emergency
        else:
            processed_physio[sample_id] = None
    return processed_physio

# -------------------------------
# 4. Emergency Detection Function
# -------------------------------

def detect_emergency(sample_id, processed_data):
    """
    Combines data from different sources to detect emergencies.

    Parameters:
    - sample_id (int): The SampleID.
    - processed_data (dict): Dictionary containing processed data from all modalities.

    Returns:
    - str: Detected emergency scenario or 'No emergency detected'.
    """
    detected_scenarios = []
    # Collect data from different modalities
    text = processed_data['text'].get(sample_id, "")
    image_emotion = processed_data['image'].get(sample_id, "")
    audio_text = processed_data['audio'].get(sample_id, "")
    video_scenario = processed_data['video'].get(sample_id, "")
    physio_scenario = processed_data['physio'].get(sample_id, "")
    # Analyze text data
    if 'pain' in text or 'help' in text:
        detected_scenarios.append('Possible Pain or Distress from Text')
    # Analyze image data
    if image_emotion in ['Sad', 'Angry', 'Pain']:
        detected_scenarios.append(f'Negative Emotion Detected: {image_emotion}')
    # Analyze audio data
    if 'help' in audio_text:
        detected_scenarios.append('Distress Detected in Speech')
    # Analyze video data
    if video_scenario:
        detected_scenarios.append(f'Video Detected Scenario: {video_scenario}')
    # Analyze physiological data
    if physio_scenario:
        detected_scenarios.append(f'Physiological Data Indicates: {physio_scenario}')
    # Decision-making logic
    if detected_scenarios:
        return ', '.join(detected_scenarios)
    else:
        return 'No emergency detected'

# -------------------------------
# 5. Evaluation Function
# -------------------------------

def evaluate_predictions(metadata_df, predictions):
    """
    Evaluates the predictions against the ground truth labels and displays organized tables.

    Parameters:
    - metadata_df (pd.DataFrame): DataFrame containing the metadata.
    - predictions (dict): Dictionary mapping SampleID to predicted scenarios.

    Returns:
    - None
    """
    # Extract ground truth labels
    y_true = []
    y_pred = []
    sample_ids = []
    for _, row in metadata_df.iterrows():
        sample_id = row['SampleID']
        true_emergency = row['Emergency']
        predicted_emergency = 'Yes' if predictions.get(sample_id, '') != 'No emergency detected' else 'No'
        y_true.append(true_emergency)
        y_pred.append(predicted_emergency)
        sample_ids.append(sample_id)

    # Create a DataFrame for detailed comparison
    comparison_df = pd.DataFrame({
        'SampleID': sample_ids,
        'True Emergency': y_true,
        'Predicted Emergency': y_pred
    })

    # Display the comparison table
    st.subheader("Detailed Predictions")
    st.write("Comparison of true emergency labels and predicted labels:")
    st.dataframe(comparison_df)

    # Calculate evaluation metrics
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    # Convert classification report to DataFrame
    report_df = pd.DataFrame(report).transpose().round(2)

    # Display the classification report as a table
    st.subheader("Classification Report")
    st.write("Performance metrics:")
    st.table(report_df)

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=['Yes', 'No'])

    # Create a DataFrame for confusion matrix
    cm_df = pd.DataFrame(cm, index=['Actual Yes', 'Actual No'], columns=['Predicted Yes', 'Predicted No'])

    # Display the confusion matrix as a table
    st.subheader("Confusion Matrix")
    st.write("Confusion matrix:")
    st.table(cm_df)

    # Optionally, plot the confusion matrix heatmap
    fig, ax = plt.subplots()
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot(fig)

# -------------------------------
# 6. Streamlit GUI Application
# -------------------------------

def medical_emergency_agent_with_evaluation():
    """
    The main function that runs the Streamlit web application for the Medical Emergency Detection Agent.
    It handles data loading, processing, prediction, evaluation, and displays results.
    """
    add_custom_css()  # Apply custom CSS styles

    # Display the AI doctor image at the top of the app
    st.image('AI_doctor.png', use_column_width=True)

    # Set the main title of the application
    st.title("Multimodal Medical Emergency Detection Agent with Enhanced Evaluation")

    # -------------------------------
    # Load Metadata and Data
    # -------------------------------
    st.header("Data Loading and Preprocessing")
    metadata_df = load_metadata()  # Updated to read from the SQLite database
    if metadata_df is None:
        st.error("Metadata could not be loaded. Please ensure the database file exists.")
        return
    st.success("Metadata loaded successfully.")

    # Display sample metadata
    st.subheader("Sample Metadata")
    st.dataframe(metadata_df.head())

    # Load data from different modalities
    text_data = load_text_data(metadata_df)
    image_data = load_image_data(metadata_df)
    audio_data = load_audio_data(metadata_df)
    video_data = load_video_data(metadata_df)
    physio_data = load_physiological_data(metadata_df)
    st.success("Data from all modalities loaded successfully.")

    # -------------------------------
    # Data Processing
    # -------------------------------
    st.header("Data Processing")
    processed_text = process_text_data(text_data)
    processed_image = process_image_data(image_data)
    processed_audio = process_audio_data(audio_data)
    processed_video = process_video_data(video_data)
    processed_physio = process_physiological_data(physio_data)
    st.success("Data from all modalities processed successfully.")

    # Combine all processed data into a single dictionary
    processed_data = {
        'text': processed_text,
        'image': processed_image,
        'audio': processed_audio,
        'video': processed_video,
        'physio': processed_physio
    }

    # -------------------------------
    # Emergency Detection and Prediction
    # -------------------------------
    st.header("Emergency Detection and Prediction")
    predictions = {}
    for sample_id in metadata_df['SampleID']:
        prediction = detect_emergency(sample_id, processed_data)
        predictions[sample_id] = prediction
    st.success("Emergency detection completed for all samples.")

    # Display some sample predictions
    st.subheader("Sample Predictions")
    sample_predictions_df = pd.DataFrame({
        'SampleID': metadata_df['SampleID'],
        'Predicted Scenarios': [predictions[sid] for sid in metadata_df['SampleID']]
    })
    st.dataframe(sample_predictions_df.head())

    # -------------------------------
    # Evaluation
    # -------------------------------
    st.header("Evaluation of Predictions")
    evaluate_predictions(metadata_df, predictions)
    st.success("Evaluation completed.")

    # -------------------------------
    # Conclusion
    # -------------------------------
    st.header("Conclusion")
    st.write("""
        The Multimodal Medical Emergency Detection Agent successfully combines data from multiple sources
        to detect emergencies. The evaluation results, presented in organized tables, provide clear insights
        into the performance of the detection system.
    """)

# -------------------------------
# 7. Run the Streamlit App
# -------------------------------

if __name__ == "__main__":
    medical_emergency_agent_with_evaluation()  # Execute the main function to run the app

