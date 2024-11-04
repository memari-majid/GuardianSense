# **GuardianSense: Autonomous Multi-Modal Emergency Detection Agent**

<img src="./AI_doctor.png" alt="AI Doctor" style="width: 100%;"/>

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Features](#features)
- [Synthetic Data Generation](#synthetic-data-generation)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview
**GuardianSense** is an AI-driven, autonomous multi-modal agent designed to assist disabled individuals by:
- Continuously monitoring their environment through multiple data inputs
- Making intelligent decisions using pretrained medical models
- Triggering real-time alerts to caregivers or authorities in emergencies

## Installation

### Prerequisites
- Python 3.9 or higher
- Conda (recommended) or pip
- Git

### 1. Clone Repository
```bash
git clone https://github.com/your-repository/guardiansense.git
cd guardiansense
```

### 2. Environment Setup

#### Using Conda (Recommended)
```bash
# Create environment from YAML
conda env create -f environment.yml

# Activate environment
conda activate guardian
```

#### Manual Installation
```bash
pip install streamlit numpy Pillow opencv-python tqdm pyttsx3
```

#### Audio Generation Dependencies (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install -y espeak espeak-ng python3-espeak
```

### 3. LLM Setup (Optional)
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start service
ollama serve

# Pull models
ollama pull llama3.2
```

## Features

### Core Capabilities
- 🎥 Real-time video/image processing
- 📝 Text and database integration
- 🗣️ Speech recognition and processing
- 🌐 URL data monitoring
- 🏥 Medical decision-making
- ⌚ Health record analysis
- 📍 GPS integration
- 🚨 Emergency triggers
- 🔄 Multi-modal integration

## Synthetic Data Generation

### Overview
Generate multi-modal training data for emergency and non-emergency scenarios using an interactive Streamlit interface.

### Launch Generator
```bash
streamlit run generate_synthetic_data_streamlit.py
```

### Available Scenarios

#### Emergency Scenarios (IDs 1-12)
- Heart Attack
- Stroke
- Fall
- Respiratory Distress
- Allergic Reaction
- Seizure
- Diabetic Emergency
- Choking
- Drowning
- Poisoning
- Severe Bleeding
- Burns

#### Non-Emergency Scenarios (IDs 13-20)
- Routine Check-up
- Mild Headache
- Common Cold
- Seasonal Allergies
- Minor Cut
- Back Pain
- Stress
- Indigestion

### Generated Data Types

#### 1. Text Data (`text_data/`)
- Scenario descriptions
- Symptom transcripts
- Patient complaints
- Format: .txt files

#### 2. Image Data (`image_data/`)
- 128x128 pixel facial expressions
- Emergency: Distressed expressions
- Non-emergency: Neutral/happy expressions
- Format: .png files

#### 3. Audio Data (`audio_data/`)
- Synthesized speech
- Emergency: Distress calls
- Non-emergency: Routine complaints
- Format: .wav files

#### 4. Video Data (`video_data/`)
- Movement animations
- Emergency: Erratic movements
- Non-emergency: Stable movements
- Format: .mp4 files

#### 5. Physiological Data (`physiological_data/`)
- Vital signs:
  - Heart rate
  - Blood pressure
  - Oxygen saturation
  - Temperature
- Format: .json files

#### 6. Database (`metadata/`)
- Sample metadata
- Scenario details
- File paths
- Format: .db file

### Using the Generator

1. **Select Scenarios**
   - Use "Select All" or choose individual scenarios
   - View emergency/non-emergency status

2. **Choose Data Types**
   - Select desired data modalities
   - Enable/disable database generation

3. **Set Parameters**
   - Number of instances (1-10000)
   - Default: 100 per scenario

4. **Generate Data**
   - Monitor progress bar
   - View status updates
   - Receive completion notification

### Output Structure
```
project_root/
├── text_data/           # Descriptions
├── image_data/          # Facial expressions
├── audio_data/          # Speech
├── video_data/          # Animations
├── physiological_data/  # Vital signs
└── metadata/           # Database
```

## Project Structure
```
guardiansense/
├── environment.yml
├── generate_synthetic_data.py
├── generate_synthetic_data_streamlit.py
├── emergency_detect.py
├── README.md
└── data/
```

## Contributing
We welcome contributions! Please feel free to:
- Submit pull requests
- Report issues
- Suggest features
- Improve documentation

## License
This project is licensed under the MIT License.

## Contact
- **Email**: mmemari@uvu.edu
- **Issues**: [GitHub Issues](https://github.com/your-repository/guardiansense/issues)
