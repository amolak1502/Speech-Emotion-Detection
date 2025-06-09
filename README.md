# Speech Emotion Detection: Comparing CNN and LSTM on SAVEE and TESS Datasets

## Overview
This project focuses on detecting emotions from speech using deep learning models, specifically Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks. We evaluate and compare the performance of these models on two datasets: SAVEE (Surrey Audio-Visual Expressed Emotion) and TESS (Toronto Emotional Speech Set). The project includes both baseline implementations and optimized versions of the models to enhance accuracy and generalization.

## Datasets
- **SAVEE**: Contains audio recordings from actors expressing seven emotions: anger, disgust, fear, happiness, sadness, surprise, and neutral.
- **TESS**: Comprises audio recordings from two actresses expressing seven emotions: anger, disgust, fear, happiness, pleasant surprise (mapped as surprise), sadness, and neutral.

## Models
- **CNN**: A Convolutional Neural Network designed to extract spatial features from audio data.
- **LSTM**: A Long Short-Term Memory network designed to capture temporal dependencies in audio sequences.

## Notebooks
- **`TESS_CNN.ipynb`**:
  - Implements a base CNN model for emotion detection on the TESS dataset.
  - Includes an optimized CNN model with Batch Normalization, Strided Convolutions instead of MaxPooling, a Squeeze-and-Excitation attention mechanism, a learning rate of 0.0009, and LeakyReLU activation.
- **`TESS_LSTM.ipynb`**:
  - Implements a base LSTM model for emotion detection on the TESS dataset.
  - Includes an optimized LSTM model with Batch Normalization, early dropout, and L2 regularization.
- **`SAVEE_CNN_LSTM.ipynb`**:
  - Implements both base CNN and LSTM models for emotion detection on the SAVEE dataset.
  - Includes an optimized LSTM model with a self-attention layer (via `keras-self-attention`).

Each notebook performs data loading, preprocessing, feature extraction (e.g., MFCC, chroma, mel-spectrogram), model training, and evaluation with metrics like accuracy, confusion matrices, and classification reports.

## Requirements
To run the notebooks, install the following Python libraries:

```bash
pip install pandas numpy librosa seaborn matplotlib scikit-learn ipython tensorflow keras-self-attention
