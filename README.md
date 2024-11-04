# **GuardianSense: Autonomous Multi-Modal Emergency Detection Agent for Disabled Persons**
<img src="./AI_doctor.png" alt="AI Doctor" style="width: 100%;"/>

## Overview  
**GuardianSense** is an AI-driven, autonomous multi-modal agent designed to assist disabled individuals by continuously monitoring their environment through various data inputs like video, image, text, speech, and URLs. By leveraging pretrained models on medical data, GuardianSense makes intelligent, autonomous decisions to detect potential emergencies and triggers alarms to notify caregivers or authorities in real-time.

## Installation Guide

### 1. Clone the Repository
```bash
git clone https://github.com/your-repository/guardiansense.git
cd guardiansense
```

### 2. Environment Setup

#### Option 1: Using Conda (Recommended)
1. Install Miniconda or Anaconda if not already installed
2. Create and activate the environment using the provided YAML file:
```bash
# Create environment from YAML
conda env create -f environment.yml

# Activate the environment
conda activate guardian
```

The `environment.yml` file includes all necessary dependencies:
```yaml
name: guardian
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - streamlit
  - numpy
  - pillow
  - opencv
  - tqdm
  - pip
  - pip:
    - pyttsx3
    - espeak
```

#### Option 2: Manual Installation
If you prefer not to use conda, install dependencies manually:
```bash
pip install streamlit numpy Pillow opencv-python tqdm pyttsx3
```

#### Additional System Dependencies
For Ubuntu/Debian systems (required for audio generation):
```bash
sudo apt-get update
sudo apt-get install -y espeak espeak-ng python3-espeak
```

### 3. LLM Setup (Optional)
If using the LLM features:

1. Install Ollama:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

2. Start Ollama service:
```bash
ollama serve
```

3. Pull required models:
```bash
ollama pull llama3.2
```

## Usage

### 1. Synthetic Data Generation
Generate multi-modal training data using either:

#### Streamlit Interface (Recommended)
```bash
streamlit run generate_synthetic_data_streamlit.py
```
Features:
- Interactive scenario selection
- Multi-modal data type options
- Real-time progress tracking
- Visual feedback

#### Command Line Interface
```bash
python generate_synthetic_data.py
```

The generator creates:
```
project_root/
├── text_data/           # Scenario descriptions
├── image_data/          # Facial expressions
├── audio_data/          # Synthesized speech
├── video_data/          # Scenario animations
├── physiological_data/  # Vital signs data
└── metadata/           # SQLite database
```

### 2. Emergency Detection
Run the main application:
```bash
streamlit run emergency_detect.py
```

## Features

### Core Capabilities
- Real-time video/image processing
- Text and database integration
- Speech recognition and processing
- URL data monitoring
- Medical decision-making
- Health record analysis
- GPS integration
- Emergency triggers
- Multi-modal integration

### Synthetic Data Generation
- Multiple emergency/non-emergency scenarios
- Multi-modal data generation
- Customizable generation options
- Progress tracking
- Database integration

## Project Structure
```
guardiansense/
├── environment.yml          # Conda environment file
├── generate_synthetic_data.py    # Data generator core
├── generate_synthetic_data_streamlit.py  # Web interface
├── emergency_detect.py      # Main detection system
├── README.md               # Documentation
└── data/                   # Generated datasets
```

## Contributing
Contributions are welcome! Please feel free to submit pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For questions or issues:
- **Email**: mmemari@uvu.edu
- **GitHub Issues**: [Project Issues Page](https://github.com/your-repository/guardiansense/issues)

## Synthetic Data Generation System

### Overview
The synthetic data generator creates multi-modal training data for both emergency and non-emergency medical scenarios. It uses Streamlit for an interactive web interface that allows users to customize data generation.

### Scenarios Available
The generator includes 20 predefined scenarios:

**Emergency Scenarios (IDs 1-12):**
- Heart Attack: Sudden blockage of blood flow to heart
- Stroke: Brain blood supply interruption
- Fall: Unintentional falling events
- Respiratory Distress: Breathing difficulties
- Allergic Reaction: Adverse immune responses
- Seizure: Uncontrolled electrical brain disturbances
- Diabetic Emergency: Blood sugar imbalances
- Choking: Airway obstruction
- Drowning: Liquid submersion cases
- Poisoning: Harmful substance exposure
- Severe Bleeding: Excessive blood loss
- Burns: Heat/chemical tissue injuries

**Non-Emergency Scenarios (IDs 13-20):**
- Routine Check-up: Regular health examinations
- Mild Headache: Minor head pain
- Common Cold: Viral infections
- Seasonal Allergies: Environmental reactions
- Minor Cut: Small lacerations
- Back Pain: Musculoskeletal discomfort
- Stress: Emotional strain
- Indigestion: Digestive discomfort

### Data Types Generated

1. **Text Data** (`text_data/`)
   - Scenario descriptions
   - Symptom transcripts
   - Patient complaints
   - Format: .txt files

2. **Image Data** (`image_data/`)
   - Facial expressions (128x128 pixels)
   - Emergency: Distressed face with frown
   - Non-emergency: Neutral/happy face
   - Format: .png files

3. **Audio Data** (`audio_data/`)
   - Synthesized speech using pyttsx3
   - Emergency: Distress calls, pain descriptions
   - Non-emergency: Routine complaints
   - Format: .wav files

4. **Video Data** (`video_data/`)
   - Simple animations showing movement
   - Emergency: Erratic/distressed movements
   - Non-emergency: Stable/calm movements
   - Format: .mp4 files

5. **Physiological Data** (`physiological_data/`)
   - Simulated vital signs:
     - Heart rate
     - Blood pressure
     - Oxygen saturation
     - Temperature
   - Format: .json files

6. **Database** (`metadata/`)
   - SQLite database containing:
     - Sample IDs
     - Scenario details
     - File paths
     - Emergency status
   - Format: .db file

### Using the Streamlit Interface

1. **Launch the Interface**
```bash
streamlit run generate_synthetic_data_streamlit.py
```

2. **Select Scenarios**
   - Use "Select All Scenarios" checkbox for all scenarios
   - Or individually select specific scenarios
   - Each scenario shows emergency/non-emergency status

3. **Choose Data Types**
   - Use "Select All Data Types" for complete dataset
   - Or select specific data types:
     - Text data
     - Image data
     - Audio data
     - Video data
     - Physiological data
     - Database entries

4. **Set Number of Instances**
   - Choose 1-10000 instances per scenario
   - Default: 100 instances
   - Total samples = Scenarios × Instances

5. **Generate Data**
   - Click "Generate Data" button
   - Monitor progress bar
   - View real-time status updates
   - Receive completion notification

### Example Generated Data Structure
```
project_root/
├── text_data/
│   ├── sample_1.txt
│   ├── sample_2.txt
│   └── ...
├── image_data/
│   ├── sample_1.png
│   ├── sample_2.png
│   └── ...
├── audio_data/
│   ├── sample_1.wav
│   ├── sample_2.wav
│   └── ...
├── video_data/
│   ├── sample_1.mp4
│   ├── sample_2.mp4
│   └── ...
├── physiological_data/
│   ├── sample_1.json
│   ├── sample_2.json
│   └── ...
└── metadata/
    └── dataset_metadata.db
```

### Data Generation Process
1. Creates necessary directories
2. Initializes database if selected
3. Generates samples for each scenario:
   - Creates unique sample ID
   - Generates selected data types
   - Updates progress bar
   - Saves metadata to database

### Error Handling
- Validates selections before generation
- Handles missing dependencies gracefully
- Provides clear error messages
- Maintains data consistency
