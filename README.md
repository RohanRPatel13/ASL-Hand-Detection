# ASL-Hand-Detection
This is a program that is able to read the ASL alphabet with a camera input. This program uses mediapipe to detect a hand and uses Google's Teachable Machine to identify each shape of the hand. A model of each letter was made using the dataCollection.py file by creating a skeleton of the hand. This was then uploaded to Teachable Machine to create the model. The sign language is then read by the ASLReader.py file and compares it to the models to determine which letter it is.
