![logo_orange_github](https://user-images.githubusercontent.com/67295703/210141583-54b227d8-c796-45c4-9857-f42537faaa0d.png)

### Prototype of application which helps newbies in strength training
### Thesis Paper - ["Strength training assistant system using computer vision"](https://github.com/CyperStone/AI-COACH/blob/main/thesis_paper.pdf) [PL]


## Project Overview
* Gathered and labelled strength training videos of dozens medium advanced gym enthusiasts which were imitating technique errors
* Designed and created a training assistant system which recognizes start/end of the exercise, counts repetitions, and detects errors in exercise techniques (over 80% accuracy for most of the events)
* Deployed the system into an application which signals detected events with visual and audible messages in real-time

![screen-gif](https://user-images.githubusercontent.com/67295703/211409049-0c68165c-6d5d-4e5b-8789-9fda4d686ece.gif)

## Getting Started
1. Clone the repository to your local machine and navigate to the project directory:
```
git clone https://github.com/CyperStone/AI-COACH.git
```
2. Create and activate new environment with Python 3.10 (example with using Anaconda):
```
conda create -n myenv python=3.10
conda activate myenv
```
3. Install required packages:
```
conda install pip
pip install -r requirements.txt
```
4. Run the application, simply using the following command:
```
python main.py
```

## User tutorial
1. Select an exercise type (currently only the back squat is supported).

2. Select a source type - live workout or recording from training.

3. In both of the above cases:
    * camera should be set at about one meter height, a few meters away from workout spot, so that the whole body is visible
    * nothing should cover the face of the person who is performing exercise
    * the person who is doing the exercise should be facing the camera perfectly
    * video stream should be of good quality (lighting, resolution).

4. When you start any training views, the training assistant system will begin tracking the person shown in the recording. The application will signal the start of the exercise with the word "START" that will appear on the screen, and with a short beep. After starting a series of repetitions, the system will check the correctness of the exercise technique. If a technical error is detected, the screen will display the name of the error and play a voice hint telling you how to eliminate it in the next repetition. The system will also count the repetitions performed, which can be seen in the lower-left corner of the device's screen and heard in the form of short messages. When the exercise is completed, the screen will say "STOP" and a short beep will be played.

## Project structure
The project is organized into the following directories and files:
* **data_utils** - directory containing auxiliary tools for dataset preparation
  * **capture_pose_from_videos.py** - program that plays back the recordings with the applied pose estimation from a selected pre-recorded reps set
  * **label_dataset.py** - program for labeling training recordings
  * **record_videos.py** - program to automatically record exercises using three cameras
* **assets** - directory containing resources used in the project
  * **images** - directory containing icons used in the app's GUI
  * **models** - directory containing the trained machine learning models used inside the training assistant system
    * **squat** - directory containing the models used in the module that supports back squat
  * **sounds** - directory containing sound files played by the application
* **AICoach.kv** - file containing the app's GUI declaration
* **config.py** - configuration file for the application and training assistant system
* **exercise_modules.py** - file containing the declaration of modules of the training assistant system
* **main.py** - the main file of the computer application
* **pose_module.py** - file containing declaration of pose estimation model (BlazePose) as a separate class
* **requirements.txt** - text file containing a list of required packages
* **thesis_paper.pdf** - thesis paper describing the whole project in detail
* **utils.py** - file containing declarations of auxiliary classes and functions used in the project
