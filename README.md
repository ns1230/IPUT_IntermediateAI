# IPUT_IntermediateAI

This repository contains Python scripts developed for an intermediate course focusing on AI and machine learning, with a particular emphasis on audio data analysis. The scripts are part of an ongoing research project aimed at exploring the relationship between audio features and physiological responses indicative of stress.

## Research Project Context

The primary goal of this research project is to use regression analysis to understand how audio features, specifically Mel Frequency Cepstral Coefficients (MFCCs), correlate with physiological indicators of stress such as heart rate and brainwave data. By analyzing these relationships, we aim to gain insights into how certain audio features might influence or reflect the stress levels experienced by listeners.

## Contents

audioToMfcc.py: This script processes audio files to extract Mel Frequency Cepstral Coefficients (MFCCs), a feature widely used in audio analysis. It includes code for reading audio files, converting them to a WAV format, and calculating MFCCs.

mergeFiles.py: A utility script designed to merge different data sources, such as brainwave data, heart rate data, and MFCCs from audio files. It includes functions for data alignment, interpolation, and aggregation, ensuring a cohesive dataset for analysis.

regression_mfccsheart.py: This script applies machine learning techniques, specifically a Random Forest Regressor, to analyze the relationship between audio features (MFCCs) and heart rate data. It includes data preprocessing, feature selection, model training, prediction, and evaluation metrics like RMSE and R2 Score.

## How to Use

Clone the repository to your local machine.
Ensure you have libraries like librosa, pandas, sklearn, seaborn, and matplotlib installed.
Run each script in a Python environment. For instance, use python audioToMfcc.py to extract MFCCs from audio files.
