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
