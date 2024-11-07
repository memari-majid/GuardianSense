# Medical Synthetic Data Generator

A streamlined tool for generating synthetic multi-modal medical data for machine learning and testing purposes.

## Overview
Generate realistic medical scenario data across multiple modalities with customizable parameters. Ideal for creating training datasets for medical AI systems, testing healthcare applications, or simulating medical scenarios.

## Features
- 🎯 20 predefined medical scenarios (emergency & non-emergency)
- 📊 Multi-modal data generation
- 🖥️ User-friendly Streamlit interface
- 📈 Real-time progress tracking
- 🗄️ Structured data organization

## Installation

### Prerequisites
- Python 3.9+
- Conda (recommended)

### Setup
```bash
# Clone repository
git clone https://github.com/your-username/medical-synthetic-data.git
cd medical-synthetic-data

# Create and activate conda environment
conda env create -f environment.yml
conda activate synthetic-data

# For audio generation (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y espeak espeak-ng python3-espeak
```

## Quick Start
```bash
streamlit run streamlit_app.py
```

## Available Scenarios

### Emergency Scenarios
1. Heart Attack
2. Stroke
3. Fall
4. Respiratory Distress
5. Allergic Reaction
6. Seizure
7. Diabetic Emergency
8. Choking
9. Drowning
10. Poisoning
11. Severe Bleeding
12. Burns

### Non-Emergency Scenarios
13. Routine Check-up
14. Mild Headache
15. Common Cold
16. Seasonal Allergies
17. Minor Cut
18. Back Pain
19. Stress
20. Indigestion

## Data Types Generated

### 1. Text Data (`text_data/`)
- Detailed scenario descriptions
- Simulated patient complaints
- Emergency/non-emergency indicators
- Format: .txt

### 2. Image Data (`image_data/`)
- 128x128px facial expressions
- Emergency: Distressed expressions
- Non-emergency: Neutral/happy expressions
- Format: .png

### 3. Audio Data (`audio_data/`)
- Synthesized speech samples
- Scenario-specific phrases
- Emergency/non-emergency tones
- Format: .wav

### 4. Video Data (`video_data/`)
- Simple movement animations
- Emergency: Erratic patterns
- Non-emergency: Stable patterns
- Format: .mp4

### 5. Physiological Data (`physiological_data/`)
```json
{
    "heart_rate": "value",
    "blood_pressure": {
        "systolic": "value",
        "diastolic": "value"
    },
    "oxygen_saturation": "value",
    "temperature": "value"
}
```
Format: .json

### 6. Metadata Database (`metadata/`)
- Sample IDs
- Scenario details
- File paths
- Emergency status
- Format: SQLite .db

## Usage Guide

1. **Launch Application**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Select Scenarios**
   - Choose specific scenarios or "Select All"
   - Mix emergency and non-emergency cases

3. **Choose Data Types**
   - Select required modalities
   - Enable/disable database generation

4. **Configure Generation**
   - Set number of instances (1-10000)
   - Default: 100 per scenario

5. **Generate Data**
   - Monitor progress bar
   - View real-time status
   - Check completion message

## Output Structure
```
generated_data/
├── text_data/
│   ├── sample_1.txt
│   └── sample_2.txt
├── image_data/
│   ├── sample_1.png
│   └── sample_2.png
├── audio_data/
│   ├── sample_1.wav
│   └── sample_2.wav
├── video_data/
│   ├── sample_1.mp4
│   └── sample_2.mp4
├── physiological_data/
│   ├── sample_1.json
│   └── sample_2.json
└── metadata/
    └── dataset_metadata.db
```

## Project Structure
```
medical-synthetic-data/
├── synthetic_data_generator.py  # Core generation logic
├── streamlit_app.py            # Web interface
├── environment.yml             # Dependencies
└── README.md                   # Documentation
```

## Contributing
Contributions welcome! Please:
- Fork the repository
- Create a feature branch
- Submit a pull request

## License
MIT License - See LICENSE file for details

## Contact
For issues and feature requests, please use the [Issue Tracker](https://github.com/your-username/medical-synthetic-data/issues)
