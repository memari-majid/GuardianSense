# generate_synthetic_data.py

# -------------------------------
# Import Necessary Libraries
# -------------------------------

import os
import random
import numpy as np
from PIL import Image, ImageDraw
import cv2
import pyttsx3
import json
import warnings
from multiprocessing import Pool, cpu_count
from itertools import repeat
from tqdm import tqdm
import sqlite3  # For SQLite database

# Suppress any warnings for cleaner output
warnings.filterwarnings("ignore")

# -------------------------------
# Global Variables and Settings
# -------------------------------

# Number of instances per scenario (adjust as needed)
NUM_INSTANCES_PER_SCENARIO = 1000  # You can increase this number as needed

# List of scenarios with details and emergency status
SCENARIOS = [
    # Emergency Scenarios
    {'id': 1, 'name': 'Heart Attack', 'description': 'A sudden blockage of blood flow to the heart muscle.', 'emergency': 'Yes'},
    {'id': 2, 'name': 'Stroke', 'description': 'Interruption of blood supply to the brain causing brain damage.', 'emergency': 'Yes'},
    {'id': 3, 'name': 'Fall', 'description': 'An event where a person unintentionally falls to the ground.', 'emergency': 'Yes'},
    {'id': 4, 'name': 'Respiratory Distress', 'description': 'Difficulty in breathing or shortness of breath.', 'emergency': 'Yes'},
    {'id': 5, 'name': 'Allergic Reaction', 'description': 'An adverse immune response to a substance.', 'emergency': 'Yes'},
    {'id': 6, 'name': 'Seizure', 'description': 'A sudden, uncontrolled electrical disturbance in the brain.', 'emergency': 'Yes'},
    {'id': 7, 'name': 'Diabetic Emergency', 'description': 'A severe imbalance of blood sugar levels.', 'emergency': 'Yes'},
    {'id': 8, 'name': 'Choking', 'description': 'Obstruction of the airway preventing normal breathing.', 'emergency': 'Yes'},
    {'id': 9, 'name': 'Drowning', 'description': 'Respiratory impairment due to submersion in liquid.', 'emergency': 'Yes'},
    {'id': 10, 'name': 'Poisoning', 'description': 'Ingestion or exposure to harmful substances.', 'emergency': 'Yes'},
    {'id': 11, 'name': 'Severe Bleeding', 'description': 'Excessive blood loss due to injury.', 'emergency': 'Yes'},
    {'id': 12, 'name': 'Burns', 'description': 'Injury to tissue caused by heat, chemicals, electricity, or radiation.', 'emergency': 'Yes'},
    # Non-Emergency Scenarios
    {'id': 13, 'name': 'Routine Check-up', 'description': 'A regular health examination.', 'emergency': 'No'},
    {'id': 14, 'name': 'Mild Headache', 'description': 'A minor pain in the head.', 'emergency': 'No'},
    {'id': 15, 'name': 'Common Cold', 'description': 'A viral infection causing sneezing and sore throat.', 'emergency': 'No'},
    {'id': 16, 'name': 'Seasonal Allergies', 'description': 'Allergic reactions to environmental triggers.', 'emergency': 'No'},
    {'id': 17, 'name': 'Minor Cut', 'description': 'A small skin laceration.', 'emergency': 'No'},
    {'id': 18, 'name': 'Back Pain', 'description': 'Discomfort in the back area.', 'emergency': 'No'},
    {'id': 19, 'name': 'Stress', 'description': 'Emotional strain or tension.', 'emergency': 'No'},
    {'id': 20, 'name': 'Indigestion', 'description': 'Discomfort in the stomach area.', 'emergency': 'No'}
]

# -------------------------------
# Function to Create Directories
# -------------------------------

def create_directories():
    """
    Creates the necessary directories for storing the generated data.
    """
    dirs = [
        'text_data',           # Directory for text data files
        'image_data',          # Directory for image data files
        'audio_data',          # Directory for audio data files
        'video_data',          # Directory for video data files
        'physiological_data',  # Directory for physiological data files
        'metadata'             # Directory for metadata files and database
    ]
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)

# -------------------------------
# Database Functions
# -------------------------------

def create_database():
    """
    Creates the SQLite database and the metadata table.
    """
    conn = sqlite3.connect('metadata/dataset_metadata.db')
    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS metadata (
            SampleID INTEGER PRIMARY KEY,
            ScenarioID INTEGER,
            ScenarioName TEXT,
            ScenarioDescription TEXT,
            Emergency TEXT,
            TextDataPath TEXT,
            ImageDataPath TEXT,
            AudioDataPath TEXT,
            VideoDataPath TEXT,
            PhysiologicalDataPath TEXT
        )
    ''')

    conn.commit()
    conn.close()

def insert_metadata(scenario_data):
    """
    Inserts a single metadata record into the database.

    Parameters:
    - scenario_data (dict): The metadata for a single sample.
    """
    conn = sqlite3.connect('metadata/dataset_metadata.db')
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO metadata (
            SampleID,
            ScenarioID,
            ScenarioName,
            ScenarioDescription,
            Emergency,
            TextDataPath,
            ImageDataPath,
            AudioDataPath,
            VideoDataPath,
            PhysiologicalDataPath
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        scenario_data['SampleID'],
        scenario_data['ScenarioID'],
        scenario_data['ScenarioName'],
        scenario_data['ScenarioDescription'],
        scenario_data['Emergency'],
        scenario_data['TextDataPath'],
        scenario_data['ImageDataPath'],
        scenario_data['AudioDataPath'],
        scenario_data['VideoDataPath'],
        scenario_data['PhysiologicalDataPath']
    ))

    conn.commit()
    conn.close()

# -------------------------------
# Data Generation Functions
# -------------------------------

def generate_text_data(scenario_data):
    """
    Generates text data for a given scenario.

    Parameters:
    - scenario_data (dict): Metadata for the current sample.

    Returns:
    - str: File path to the generated text data.
    """
    scenario = scenario_data['ScenarioName']   # Get the scenario name
    sample_id = scenario_data['SampleID']      # Get the sample ID

    # Define text phrases for different scenarios
    text_phrases = {
        # Emergency Scenarios
        'Heart Attack': [
            "I'm experiencing severe chest pain.",
            "It feels like there's pressure on my chest.",
            "My left arm is numb.",
            "I have a crushing sensation in my chest."
        ],
        'Stroke': [
            "I can't move my arm.",
            "My face feels droopy.",
            "I'm having trouble speaking.",
            "I have a sudden headache."
        ],
        'Severe Bleeding': [
            "I'm bleeding heavily.",
            "I can't stop the bleeding.",
            "There's a lot of blood."
        ],
        'Burns': [
            "I burned my hand badly.",
            "My skin is blistering.",
            "It's a severe burn."
        ],
        # Non-Emergency Scenarios
        'Routine Check-up': [
            "I'm here for my annual physical.",
            "Just a regular check-up.",
            "No specific complaints, just a routine visit."
        ],
        'Mild Headache': [
            "I have a slight headache.",
            "My head hurts a little.",
            "It's a mild pain in my head."
        ],
        'Common Cold': [
            "I've been sneezing a lot.",
            "I have a runny nose.",
            "I think I caught a cold."
        ],
        # Additional phrases for other scenarios...
    }

    # Select random phrases for the current scenario
    text = ' '.join(random.choices(
        text_phrases.get(scenario, ["No specific complaint."]),
        k=random.randint(1, 3)
    ))

    filename = f'text_data/sample_{sample_id}.txt'  # Define the filename

    # Write the text data to the file
    with open(filename, 'w') as f:
        f.write(text)

    return filename  # Return the path to the text file

def generate_image_data(scenario_data):
    """
    Generates image data (simplified facial expressions) for a given scenario.

    Parameters:
    - scenario_data (dict): Metadata for the current sample.

    Returns:
    - str: File path to the generated image data.
    """
    scenario = scenario_data['ScenarioName']   # Get the scenario name
    sample_id = scenario_data['SampleID']      # Get the sample ID

    # Create a new white image of size 128x128 pixels
    img = Image.new('RGB', (128, 128), color='white')
    draw = ImageDraw.Draw(img)  # Create a drawing context

    # Simplified representation of facial expressions
    if scenario_data['Emergency'] == 'Yes':
        # Draw a distressed face for emergency scenarios
        draw.ellipse((32, 32, 96, 96), fill=(255, 224, 189))    # Face
        draw.ellipse((50, 50, 58, 58), fill='black')            # Left eye
        draw.ellipse((70, 50, 78, 58), fill='black')            # Right eye
        draw.arc((50, 70, 78, 90), start=180, end=360, fill='red', width=3)  # Frowning mouth
    else:
        # Draw a neutral or happy face for non-emergency scenarios
        draw.ellipse((32, 32, 96, 96), fill=(255, 224, 189))    # Face
        draw.ellipse((50, 50, 58, 58), fill='black')            # Left eye
        draw.ellipse((70, 50, 78, 58), fill='black')            # Right eye
        draw.arc((50, 70, 78, 90), start=0, end=180, fill='green', width=3)  # Smiling mouth

    filename = f'image_data/sample_{sample_id}.png'  # Define the filename

    # Save the image to the specified file
    img.save(filename)

    return filename  # Return the path to the image file

def generate_audio_data(scenario_data):
    """
    Generates audio data (speech) for a given scenario using pyttsx3.

    Parameters:
    - scenario_data (dict): Metadata for the current sample.

    Returns:
    - str: File path to the generated audio data.
    """
    scenario = scenario_data['ScenarioName']   # Get the scenario name
    sample_id = scenario_data['SampleID']      # Get the sample ID

    # Define audio phrases for different scenarios
    audio_phrases = {
        # Emergency Scenarios
        'Heart Attack': [
            "Please help, my chest hurts!",
            "I think I'm having a heart attack.",
            "I can't breathe properly.",
            "Call an ambulance!"
        ],
        'Stroke': [
            "I can't move my arm.",
            "My face feels droopy.",
            "I'm having trouble speaking.",
            "I have a sudden headache."
        ],
        'Severe Bleeding': [
            "I can't stop the bleeding!",
            "There's too much blood.",
            "I need immediate help."
        ],
        'Burns': [
            "I burned myself badly.",
            "It hurts a lot.",
            "My skin is burnt."
        ],
        # Non-Emergency Scenarios
        'Routine Check-up': [
            "I'm here for a regular check-up.",
            "Just a routine visit.",
            "No urgent issues."
        ],
        'Mild Headache': [
            "I have a slight headache.",
            "It's not too bad, just annoying.",
            "I took some painkillers."
        ],
        'Common Cold': [
            "I've been feeling under the weather.",
            "Just a bit of a cold.",
            "Nothing serious, just a runny nose."
        ],
        # Additional phrases for other scenarios...
    }

    # Select a random phrase for the current scenario
    text = random.choice(
        audio_phrases.get(scenario, ["No specific complaint."])
    )

    filename = f'audio_data/sample_{sample_id}.wav'  # Define the filename

    # Initialize the pyttsx3 engine
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)    # Speech rate (words per minute)
    engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)

    # Save the speech audio to a file
    engine.save_to_file(text, filename)
    engine.runAndWait()
    engine.stop()

    return filename  # Return the path to the audio file

def generate_video_data(scenario_data):
    """
    Generates video data (simple animations) for a given scenario.

    Parameters:
    - scenario_data (dict): Metadata for the current sample.

    Returns:
    - str: File path to the generated video data.
    """
    scenario = scenario_data['ScenarioName']
    sample_id = scenario_data['SampleID']

    frame_width = 128
    frame_height = 128
    num_frames = random.randint(15, 30)  # Random number of frames

    # Output video filename
    output_filename = f'video_data/sample_{sample_id}.mp4'

    # Initialize video writer with specified parameters
    out = cv2.VideoWriter(
        output_filename,
        cv2.VideoWriter_fourcc(*'mp4v'),
        15,
        (frame_width, frame_height)
    )

    # Loop through each frame
    for frame_num in range(num_frames):
        # Create a blank frame (black image)
        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        if scenario_data['Emergency'] == 'Yes':
            # Simulate distress by random movement
            x_pos = frame_width // 2 + random.randint(-5, 5)
            y_pos = frame_height // 2 + random.randint(-5, 5)
            color = (0, 0, 255)  # Red color for emergency
        else:
            # Simulate calmness with stationary object
            x_pos = frame_width // 2
            y_pos = frame_height // 2
            color = (0, 255, 0)  # Green color for non-emergency

        # Draw a circle representing a person/object
        cv2.circle(frame, (x_pos, y_pos), 10, color, -1)

        # Write the frame to the video file
        out.write(frame)

    # Release the video writer
    out.release()

    return output_filename  # Return the path to the video file

def generate_physiological_data(scenario_data):
    """
    Generates physiological data for a given scenario.

    Parameters:
    - scenario_data (dict): Metadata for the current sample.

    Returns:
    - str: File path to the generated physiological data.
    """
    scenario = scenario_data['ScenarioName']
    sample_id = scenario_data['SampleID']
    emergency = scenario_data['Emergency']

    if emergency == 'Yes':
        # Simulate abnormal vital signs for emergency scenarios
        heart_rate = random.randint(90, 150)
        systolic_bp = random.randint(80, 180)
        diastolic_bp = random.randint(50, 110)
        oxygen_saturation = random.uniform(70, 95)
        temperature = random.uniform(97, 103)
    else:
        # Normal vital signs for non-emergency scenarios
        heart_rate = random.randint(60, 100)
        systolic_bp = random.randint(110, 130)
        diastolic_bp = random.randint(70, 85)
        oxygen_saturation = random.uniform(95, 100)
        temperature = random.uniform(97, 99)

    # Create a dictionary with the physiological data
    data = {
        'SampleID': sample_id,
        'HeartRate': heart_rate,
        'SystolicBP': systolic_bp,
        'DiastolicBP': diastolic_bp,
        'OxygenSaturation': round(oxygen_saturation, 1),
        'Temperature': round(temperature, 1),
        'ScenarioName': scenario,
        'ScenarioDescription': scenario_data['ScenarioDescription'],
        'Emergency': emergency
    }

    filename = f'physiological_data/sample_{sample_id}.json'  # Define the filename

    # Save the physiological data to a JSON file
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    return filename  # Return the path to the JSON file

# -------------------------------
# Function to Generate One Sample
# -------------------------------

def generate_sample(scenario_data):
    """
    Generates all data modalities for one sample and updates scenario_data with file paths.
    Also inserts the metadata into the database.

    Parameters:
    - scenario_data (dict): Metadata for the current sample.

    Returns:
    - dict: Updated scenario_data with file paths.
    """
    # Generate data for each modality
    text_path = generate_text_data(scenario_data)
    image_path = generate_image_data(scenario_data)
    audio_path = generate_audio_data(scenario_data)
    video_path = generate_video_data(scenario_data)
    physio_path = generate_physiological_data(scenario_data)

    # Update scenario_data with paths to the generated data files
    scenario_data.update({
        'TextDataPath': text_path,
        'ImageDataPath': image_path,
        'AudioDataPath': audio_path,
        'VideoDataPath': video_path,
        'PhysiologicalDataPath': physio_path
    })

    # Insert the metadata into the database
    insert_metadata(scenario_data)

    return scenario_data

# -------------------------------
# Main Function for Parallel Processing
# -------------------------------

def main():
    """
    Main function to execute the data generation process with parallel processing.
    """
    create_directories()
    create_database()

    sample_id_counter = 0
    tasks = []

    # Prepare tasks for multiprocessing
    for scenario in SCENARIOS:
        for instance_num in range(NUM_INSTANCES_PER_SCENARIO):
            sample_id_counter += 1
            scenario_data = {
                'SampleID': sample_id_counter,
                'ScenarioID': scenario['id'],
                'ScenarioName': scenario['name'],
                'ScenarioDescription': scenario['description'],
                'Emergency': scenario['emergency']
            }
            tasks.append(scenario_data)

    # Determine the number of processes to use
    num_processes = min(cpu_count(), 8)  # Adjust as needed

    # Use multiprocessing Pool to generate samples in parallel
    with Pool(processes=num_processes) as pool:
        # Use tqdm for progress bar
        list(tqdm(pool.imap_unordered(generate_sample, tasks), total=len(tasks)))

    print("Data generation completed successfully.")

# -------------------------------
# Entry Point of the Script
# -------------------------------

if __name__ == '__main__':
    main()
