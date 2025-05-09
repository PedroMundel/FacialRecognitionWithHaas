# Facial Recognition with HAAS

This project implements a facial recognition system using Haar Cascade classifiers in Python, leveraging OpenCV for real-time face, eye, nose, and mouth detection. It captures facial data, trains a model to recognize individuals, and allows real-time recognition via a webcam feed.

## Features

- **Real-time Face Detection**: Detect faces and specific facial features (eyes, nose, mouth) from live video.
- **Face Recognition**: Train a model to recognize users based on previously captured images.
- **Customizable**: Easily extendable with additional classifiers or features.

## Technologies Used

- Python 3.x
- OpenCV
- Haar Cascade Classifiers for Face, Eye, Nose, and Mouth Detection

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/PedroMundel/FacialRecognitionWithHaas.git
2. Install the required dependencies:
   opencv-python, opencv-contrib-python, numpy, PIL

3. Create a folder named 'data' and run generateDataset.py
   
4. Run trainClassifier.py

5. Run the application:
   python main.py

## File Structure:

- **main.py**: Main script to run the facial recognition application.

- **generateDataset.py**: Script to collect face data and store it for training.

- **train_classifier.py**: Script to train a face recognition model using the captured data.

- **haarcascade_mcs_nose.xml, haarcascade_mcs_mouth.xml**: Haar Cascade Classifiers used for detecting the nose and mouth features.


## Acknowledgements

This project uses Haar Cascade Classifiers for detecting facial features, which were developed by:

Castrillón Santana, M., Déniz Suárez, O., Hernández Tejera, M., & Guerra Artal, C. (2007) in their paper "ENCARA2: Real-time Detection of Multiple Faces at Different Resolutions in Video Streams", Journal of Visual Communication and Image Representation.

Additionally, these classifiers were trained using OpenCV's CascadeClassifier API.

## License:

This project is an academic assignment from the Universidade Federal de Goiás (UFG). This code and it's content are restricted for educational purposes, unless otherwise specified.
