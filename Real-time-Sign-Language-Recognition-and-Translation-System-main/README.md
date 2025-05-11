# Real time Sign Language Recognition and Translation System 🤚🌐

This project utilizes computer vision and machine learning to recognize hand gestures representing letters and phrases in American Sign Language (ASL), and translates them into text. Additionally, it offers translation capabilities into multiple languages, including Indian languages like Hindi, Telugu, Tamil, Kannada, and Malayalam.

## Prerequisites 🛠️

Before running this project, ensure you have the following installed:

- Python (version 3.6 or higher) 🐍
- OpenCV (`opencv-python`) 📷
- Mediapipe (`mediapipe`) 🖐️
- Google Translate API (`googletrans==4.0.0-rc1`) 🌍
- Spellchecker (`spellchecker`) ✏️

You can install these dependencies using Python's package manager (`pip`):

```
pip install opencv-python mediapipe googletrans==4.0.0-rc1 spellchecker
```

## Project Structure 📂

The project is divided into several components:

1. **Data Collection (`collect_images.py`)**:
   - Captures images from the camera for each letter of the alphabet and additional messages like "Thank You" and "Nice To Meet You".
   - Images are saved in directories corresponding to their labels in `./data`.

2. **Data Preparation (`create_dataset.py`)**:
   - Processes the collected images using Mediapipe to extract hand landmarks.
   - Constructs a dataset (`data.pickle`) containing landmark data and their corresponding labels.

3. **Model Training (`train_model.py`)**:
   - Loads the dataset from `data.pickle`.
   - Trains a RandomForestClassifier to classify hand gestures based on extracted landmarks.
   - Evaluates and saves the trained model (`model.p`).

4. **Real-time Translation and GUI (`translate_gui.py`)**:
   - Implements real-time hand gesture recognition using the trained model and Mediapipe.
   - Displays recognized gestures as text in English.
   - Provides translation of recognized text into selected languages via Google Translate API.

## Usage 🚀

1. **Data Collection**:
   - Run `collect_images.py` to capture images for training. Follow on-screen instructions to capture gestures and messages.

2. **Data Preparation**:
   - Execute `create_dataset.py` to process captured images and create `data.pickle`.

3. **Model Training**:
   - Run `train_model.py` to train the RandomForestClassifier using the prepared dataset.

4. **Real-time Translation and GUI**:
   - Launch `translate_gui.py` to open the graphical user interface.
   - Select a language from the dropdown menu to translate recognized gestures.

## Notes ℹ️

- Ensure proper lighting and hand positioning while collecting images for better recognition accuracy.
- The GUI updates in real-time based on camera input. Close the GUI window to terminate the program.
- Adjust parameters and add more gestures as needed for your specific application.
