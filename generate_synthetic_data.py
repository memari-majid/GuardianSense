# generate_synthetic_data.py

# -------------------------------
# Import Necessary Libraries
# -------------------------------

import os
import random
import numpy as np
from PIL import Image, ImageDraw
import cv2
import json
import warnings
from multiprocessing import Pool, cpu_count
from itertools import repeat
from tqdm import tqdm
import sqlite3  # For SQLite database
from functools import partial

# Optional imports with error handling
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None
    print("Warning: pyttsx3 not installed. Audio generation will be disabled.")

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
    If pyttsx3 is not available, returns a placeholder file path.
    """
    if pyttsx3 is None:
        # Return a placeholder if pyttsx3 is not available
        sample_id = scenario_data['SampleID']
        filename = f'audio_data/sample_{sample_id}.wav'
        with open(filename, 'w') as f:
            f.write("Audio generation disabled - pyttsx3 not installed")
        return filename

    scenario = scenario_data['ScenarioName']
    sample_id = scenario_data['SampleID']

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

def generate_sample(scenario_data, selected_types):
    """
    Generates selected data modalities for one sample and updates scenario_data with file paths.
    
    Parameters:
    - scenario_data (dict): Metadata for the current sample
    - selected_types (dict): Dictionary indicating which data types to generate
    
    Returns:
    - dict: Updated scenario_data with file paths
    """
    paths = {}
    
    # Generate only selected data types
    if selected_types['text']:
        paths['TextDataPath'] = generate_text_data(scenario_data)
    
    if selected_types['image']:
        paths['ImageDataPath'] = generate_image_data(scenario_data)
    
    if selected_types['audio']:
        paths['AudioDataPath'] = generate_audio_data(scenario_data)
    
    if selected_types['video']:
        paths['VideoDataPath'] = generate_video_data(scenario_data)
    
    if selected_types['physiological']:
        paths['PhysiologicalDataPath'] = generate_physiological_data(scenario_data)
    
    # Update scenario_data with the generated paths
    scenario_data.update(paths)
    
    # Insert into database if selected
    if selected_types['database']:
        insert_metadata(scenario_data)
    
    return scenario_data

# -------------------------------
# Main Function for Parallel Processing
# -------------------------------

def main():
    """
    Main function to execute the data generation process with user selections.
    """
    print("Welcome to the Synthetic Medical Data Generator!")
    
    # Get user selections
    selected_scenarios = get_user_scenarios()
    num_instances = get_num_instances()
    selected_types = get_user_data_types()
    
    # Create only necessary directories
    dirs = []
    if selected_types['text']: dirs.append('text_data')
    if selected_types['image']: dirs.append('image_data')
    if selected_types['audio']: dirs.append('audio_data')
    if selected_types['video']: dirs.append('video_data')
    if selected_types['physiological']: dirs.append('physiological_data')
    if selected_types['database']: dirs.append('metadata')
    
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
    
    # Create database if needed
    if selected_types['database']:
        create_database()
    
    # Prepare tasks for multiprocessing
    tasks = []
    sample_id_counter = 0
    
    for scenario in selected_scenarios:
        for instance_num in range(num_instances):
            sample_id_counter += 1
            scenario_data = {
                'SampleID': sample_id_counter,
                'ScenarioID': scenario['id'],
                'ScenarioName': scenario['name'],
                'ScenarioDescription': scenario['description'],
                'Emergency': scenario['emergency']
            }
            tasks.append((scenario_data, selected_types))
    
    # Use multiprocessing Pool to generate samples in parallel
    num_processes = min(cpu_count(), 8)
    with Pool(processes=num_processes) as pool:
        # Use partial to pass selected_types to generate_sample
        generate_func = partial(generate_sample, selected_types=selected_types)
        list(tqdm(pool.imap_unordered(generate_func, [t[0] for t in tasks]), 
                 total=len(tasks),
                 desc="Generating data"))
    
    print("\nData generation completed successfully!")
    print(f"Generated {len(tasks)} samples for {len(selected_scenarios)} scenarios.")

# -------------------------------
# Entry Point of the Script
# -------------------------------

if __name__ == '__main__':
    main()

# Add these functions after the global variables and before the data generation functions

def get_user_scenarios():
    """
    Allows users to select which scenarios they want to generate data for.
    
    Returns:
    - list: Selected scenarios
    """
    print("\nAvailable Scenarios:")
    print("-------------------")
    for scenario in SCENARIOS:
        print(f"{scenario['id']}. {scenario['name']} (Emergency: {scenario['emergency']})")
    
    while True:
        try:
            selection = input("\nEnter scenario IDs (comma-separated) or 'all' for all scenarios: ").strip()
            if selection.lower() == 'all':
                return SCENARIOS
            
            selected_ids = [int(id.strip()) for id in selection.split(',')]
            selected_scenarios = [s for s in SCENARIOS if s['id'] in selected_ids]
            
            if not selected_scenarios:
                print("No valid scenarios selected. Please try again.")
                continue
                
            return selected_scenarios
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas or 'all'.")

def get_user_data_types():
    """
    Allows users to select which types of data they want to generate.
    
    Returns:
    - dict: Selected data types with boolean values
    """
    data_types = {
        'text': 'Text data (descriptions and transcripts)',
        'image': 'Image data (facial expressions)',
        'audio': 'Audio data (synthesized speech)',
        'video': 'Video data (simple animations)',
        'physiological': 'Physiological data (vital signs)',
        'database': 'Database entries (metadata)'
    }
    
    print("\nAvailable Data Types:")
    print("--------------------")
    for key, description in data_types.items():
        print(f"- {key}: {description}")
    
    while True:
        selection = input("\nEnter data types (comma-separated) or 'all' for all types: ").strip()
        if selection.lower() == 'all':
            return {k: True for k in data_types}
        
        try:
            selected_types = [t.strip().lower() for t in selection.split(',')]
            result = {k: (k in selected_types) for k in data_types}
            
            if not any(result.values()):
                print("No valid data types selected. Please try again.")
                continue
                
            return result
        except ValueError:
            print("Invalid input. Please enter data types separated by commas or 'all'.")

def get_num_instances():
    """
    Allows users to specify the number of instances per scenario.
    
    Returns:
    - int: Number of instances per scenario
    """
    while True:
        try:
            num = int(input("\nEnter number of instances per scenario (1-10000): "))
            if 1 <= num <= 10000:
                return num
            print("Please enter a number between 1 and 10000.")
        except ValueError:
            print("Invalid input. Please enter a number.")
