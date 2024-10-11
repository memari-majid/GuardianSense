# **GuardianSense: Autonomous Multi-Modal Emergency Detection Agent for Disabled Persons**
![GuardianSense AI](./AI.png)
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
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Step 2: Install Package Manager
To manage Python environments more effectively, we recommend using **Miniconda**. Follow these steps to install Miniconda:

1. **Download Miniconda**: Go to the [Miniconda Installation page](https://docs.conda.io/en/latest/miniconda.html) and download the installer suitable for your operating system (Windows, macOS, Linux).

2. **Run the Installer**:
   - On **Windows**, open the installer and follow the installation instructions.
   - On **macOS** and **Linux**, run the following commands in your terminal (replace the installer name with the appropriate one for your OS):

   For Linux:
   ```bash
   bash Miniconda3-latest-Linux-x86_64.sh
   ```

   For macOS:
   ```bash
   bash Miniconda3-latest-MacOSX-x86_64.sh
   ```

3. **Create a New Conda Environment with Python 3.9**  
Once Miniconda is installed, create a new environment with Python 3.9 using the following commands:
```bash
conda create --name guardian python=3.9
```
Activate the environment:
```bash
conda activate guardian
```

### Step 3: Install Python Dependencies
Install the required Python packages by running the following command:
```bash
pip install -r requirements.txt
```

### Step 4: Install Ollama  
Ollama is the core platform for running the local language models.

#### macOS
[Download](https://ollama.com/download/Ollama-darwin.zip)

#### Windows
[Download](https://ollama.com/download/OllamaSetup.exe)

#### Linux Install
To install Ollama, run the following command:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Alternatively, for manual installation:
```bash
curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz
sudo tar -C /usr -xzf ollama-linux-amd64.tgz
```
Start Ollama:
```bash
ollama serve
```

#### Docker  
The official [Ollama Docker image](https://hub.docker.com/r/ollama/ollama) is available on Docker Hub.

### Step 5: Pull the Required Models
You will need to download the specific models that power the RAG system.

Use the following command to pull the necessary models:
```bash
ollama pull llama3.2
```

### Running GuardianSense
To start monitoring, simply run:
```bash
python main.py
```

When `main.py` is executed, a file explorer will open, allowing you to select any data type, such as video, image, text, JSON, or audio files for analysis.

### Testing the System
You can test the agent with pre-recorded videos or text inputs by specifying a file path:
```bash
python main.py --input test_video.mp4
```

## Technologies Used
- **Python**: Main programming language.
- **OpenCV**: For real-time video processing.
- **SpeechRecognition**: For converting speech to text.
- **YOLO**: For object detection in images and videos.
- **Natural Language Processing (NLP)**: For text and speech analysis.
- **Pretrained Medical Models**: For autonomous decision-making based on medical data.
- **Apple HealthKit API**: For analyzing health records from Apple Watches.
- **GPS Integration**: For analyzing GPS data from airbags or other sources.
- **SQL**: For interacting with the database to access patient records or care instructions.

## Contributing  
Feel free to contribute to this project! If you find a bug or have a feature request, please open an issue. We also welcome pull requests for improving the project.

## License  
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact  
For any questions or issues, please reach out to the project maintainer:  
- **Email**: mmemari@uvu.edu
