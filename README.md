# **GuardianSense: Autonomous Multi-Modal Emergency Detection Agent for Disabled Persons**
<img src="./AI_doctor.png" alt="AI Doctor" style="width: 100%;"/>


## Overview  
**GuardianSense** is an AI-driven, autonomous multi-modal agent designed to assist disabled individuals by continuously monitoring their environment through various data inputs like video, image, text, speech, and URLs. By leveraging pretrained models on medical data, GuardianSense makes intelligent, autonomous decisions to detect potential emergencies and triggers alarms to notify caregivers or authorities in real-time. It also integrates health records from Apple Watches and GPS data from vehicle airbags to provide comprehensive monitoring, offering a proactive approach to enhancing the safety and independence of individuals with disabilities.

## Features  
- **Real-time video and image processing**: Detects unusual behavior, falls, or dangerous objects in the environment through video feeds and images.
- **Text and database integration**: Analyzes textual data from medical databases or personalized care instructions to assess health risks or changes in routine.
- **Speech recognition and processing**: Understands voice commands and verbal cues from users, providing an additional modality for detecting distress signals.
- **URL processing**: Monitors relevant web data sources or APIs that may provide critical information, such as weather alerts or health data.
- **Medical Decision-Making**: Uses pretrained models on medical data to autonomously assess user health and environmental risks, making decisions to trigger appropriate responses.
- **Health Record Analysis**: Analyzes health data from Apple Watches to monitor vital signs and detect anomalies in real-time.
- **GPS Integration**: Uses GPS data from vehicle airbags to detect vehicle-related emergencies or collisions.
- **File Selection Explorer**: When running `main.py`, a file explorer opens, allowing users to select any data type (e.g., video, image, text, JSON, or audio files) for analysis.
- **Emergency triggers**: Automatically triggers alarms or sends notifications in case of emergency events, alerting caregivers or emergency services.
- **Multi-modal integration**: Combines and processes data from multiple sources to form a holistic understanding of the user's environment and safety status.

## Getting Started  

### Installation

Clone this repository:
```bash
git clone https://github.com/your-repository/guardiansense.git
cd guardiansense
```

### Install Python Dependencies
Install the required Python packages by running the following command:
```bash
conda env create -f environment.yml
conda activate llm
```

### Install Ollama  
Ollama is the core platform for running the local language models.


#### Linux Install
To install Ollama, run the following command:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Start Ollama:
```bash
ollama serve
```

### Pull the Required Models
You will need to download the specific models that power the RAG system.

Use the following command to pull the necessary models:
```bash
ollama pull llama3.2
```

To run the application using Streamlit:
```bash
streamlit run main.py
```

When main.py is executed, a file explorer will open, allowing you to select any data type, such as video, image, text, JSON, or audio files for analysis.


## Technologies Used
- **Python**: Main programming language.
- **OpenCV**: For real-time video processing.
- **SpeechRecognition**: For converting speech to text.
- **YOLO**: For object detection in images and videos.
- **Natural Language Processing (NLP)**: For text and speech analysis.
- **Pretrained Medical Models**: For autonomous decision-making based on medical data.
- **Apple HealthKit API**: For analyzing health records from Apple Watches.
- **GPS Integration**: For analyzing GPS data from Apple Airtags or other sources.
- **SQL**: For interacting with the database to access patient records or care instructions.

## Contributing  
Feel free to contribute to this project! If you find a bug or have a feature request, please open an issue. We also welcome pull requests for improving the project.

## License  
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact  
For any questions or issues, please reach out to the project maintainer:  
- **Email**: mmemari@uvu.edu
