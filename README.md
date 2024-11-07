# Synthetic Medical Data Generator

A comprehensive tool for generating synthetic multi-modal medical data, including emergency and non-emergency scenarios.

## Overview
This project is a focused branch of GuardianSense, specifically handling synthetic data generation. It creates realistic medical scenario data across multiple modalities:
- Text descriptions
- Images (facial expressions)
- Audio (synthesized speech)
- Video (animations)
- Physiological data
- Metadata database

## Installation

```bash
# Clone repository
git clone https://github.com/your-username/synthetic-medical-data-generator.git
cd synthetic-medical-data-generator

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate synthetic-data
```

## Quick Start

```bash
# Launch Streamlit interface
streamlit run streamlit_app.py
```

## Generated Data Types

### 1. Text Data
- Scenario descriptions
- Patient complaints
- Symptom transcripts

### 2. Image Data
- Facial expressions
- 128x128 pixel images
- Emergency/non-emergency indicators

### 3. Audio Data
- Synthesized speech
- Emergency calls
- Patient descriptions

### 4. Video Data
- Simple animations
- Movement patterns
- Emergency indicators

### 5. Physiological Data
- Vital signs
- Medical measurements
- Time series data

### 6. Metadata
- Sample information
- Scenario details
- Data relationships

## Usage Guide

1. Launch the Streamlit interface
2. Select desired scenarios
3. Choose data types to generate
4. Set number of instances
5. Click generate
6. Monitor progress
7. Access generated data in respective folders

## Project Structure
```
synthetic-medical-data-generator/
├── synthetic_data_generator.py  # Core generation logic
├── streamlit_app.py            # Web interface
├── environment.yml             # Dependencies
└── README.md                   # Documentation
```

## Contributing
This is a focused branch for synthetic data generation. For feature requests or issues, please use the issue tracker.

## License
MIT License

## Original Project
This is a specialized branch of [GuardianSense](https://github.com/your-username/guardiansense)
