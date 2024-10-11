# **GuardianSense: Autonomous Multi-Modal Emergency Detection Agent for Disabled Persons**

## Overview  
**GuardianSense** is an AI-driven, autonomous multi-modal agent designed to assist disabled individuals by continuously monitoring their environment through various data inputs like video, image, text, speech, and URLs. Leveraging pretrained models on medical data, GuardianSense makes intelligent and autonomous decisions to detect potential emergencies and trigger alarms to notify caregivers or authorities in real-time. It also integrates health records from Apple Watches and GPS data from vehicle airbags to provide comprehensive monitoring, offering a proactive approach to enhancing the safety and independence of individuals with disabilities.

## Features  
- **Real-time video and image processing**: Detects unusual behavior, falls, or dangerous objects in the environment through video feeds and images.
- **Text and database integration**: Analyzes textual data from medical databases or personalized care instructions to assess health risks or changes in routine.
- **Speech recognition and processing**: Understands voice commands and verbal cues from users, providing an additional modality for detecting distress signals.
- **URL processing**: Monitors relevant web data sources or APIs that may provide critical information (e.g., weather alerts, health data).
- **Medical Decision-Making**: Uses pretrained models on medical data to autonomously assess user health and environmental risks, making decisions to trigger appropriate responses.
- **Health Record Analysis**: Analyzes health data from Apple Watches to monitor vital signs and detect anomalies in real-time.
- **GPS Integration**: Uses GPS data from vehicle airbags to detect vehicle-related emergencies or collisions.
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

### Configuration  
- Ensure that your environment has access to the necessary hardware (cameras, microphones, Apple Watch data, GPS sensors) and database connections.
- Modify the `config.yaml` file to include your API keys, database credentials, and paths for input data such as video streams or images.

### Running GuardianSense
To start monitoring, simply run:
```bash
python main.py
```

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
