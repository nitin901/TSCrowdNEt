# TSCrowdNEt
TS-CrowdNet: A Temporal-Spatial Framework for Crowd Counting
This repository contains the official Keras/TensorFlow implementation of TS-CrowdNet, a deep learning model designed for crowd counting in video sequences. The model leverages both spatial and temporal information to provide accurate and stable crowd count predictions.

This implementation uses a pre-trained ResNet50 model as a spatial feature extractor and an LSTM (Long Short-Term Memory) network to analyze the temporal relationship between consecutive frames.

Features
Temporal-Spatial Analysis: Processes sequences of frames instead of single images to understand crowd dynamics and motion.

Transfer Learning: Utilizes a pre-trained ResNet50 backbone to leverage powerful, generalized features learned from the ImageNet dataset.

Custom Data Generator: Includes a memory-efficient data generator that creates and feeds video sequences to the model on-the-fly.

End-to-End Training: A complete, runnable script for training, validation, and evaluation.

Performance Evaluation: Automatically calculates and displays key regression metrics, including Mean Absolute Error (MAE), Mean Squared Error (MSE), and a custom accuracy score.

Methodology
The TS-CrowdNet architecture is composed of three main stages:

Spatial Feature Extraction: A TimeDistributed ResNet50 backbone processes each frame in an input sequence to extract high-level spatial features.

Temporal Modeling: An LSTM layer receives the sequence of feature vectors from the backbone and learns the temporal patterns and dependencies between them.

Regression Head: A final set of dense layers takes the output of the LSTM and regresses a final, single value representing the predicted crowd count.

Dataset
This model is designed to be trained on the Mall Dataset.

Download Link: https://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html

Directory Structure
To run the code, you must organize your files as follows:

TS-CrowdNet/
│
├── tscrowdnet_train.py       # The main Python script
├── labels.csv                # The labels file from the dataset
└── frames/
    └── frames/
        ├── seq_000001.jpg
        ├── seq_000002.jpg
        └── ...

Installation
Clone this repository:

git clone [https://github.com/your-username/TS-CrowdNet.git](https://github.com/nitin901/TSCrowdNEt/tree/main)
cd TS-CrowdNet

Install the required Python packages. It is recommended to use a virtual environment.

pip install -r requirements.txt

A requirements.txt file should contain:

tensorflow
pandas
numpy
scipy
matplotlib
seaborn
scikit-learn

Usage
To train and evaluate the TS-CrowdNet model, simply run the main script from your terminal:

python tscrowdnet_train.py

The script will automatically:

Load and preprocess the data.

Build the TS-CrowdNet model.

Train the model for 50 epochs.

Display plots for training loss and validation predictions.

Print a final summary of the performance metrics (MAE, MSE, and Accuracy).

Results
After a full training run, the script will output:

A plot of the training and validation loss over epochs.

A scatter plot comparing the model's predicted counts to the true counts on the validation set.

A final summary in the console with the calculated MAE, MSE, and accuracy within a 20% tolerance.
