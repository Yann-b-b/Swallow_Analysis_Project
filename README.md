# Swallow Analysis Project

This repository contains the code and resources developed for the analysis of swallowing sounds using machine learning techniques. The goal of the project is to classify swallowing events and detect potential swallowing disorders by analyzing audio data from surface electromyography (SeMG) signals.

## Presentation and Final Results

[embed]https://github.com/Yann-b-b/Swallow_Analysis_Project/blob/main/SRC%20Presentation%20(5).pdf[embed]

## Project Structure

The repository is structured as follows:

### Notebooks

- **Preprocessing1_SeMG_Dataset_Output_is_CSV.ipynb**  
  This notebook is responsible for preprocessing the SeMG dataset and extracting all relevant information (contact microphone and labels) from it into CSV format. It handles the extraction and labeling of swallow events from the raw SeMG data.

- **Preprocessing2_Convert_All_SeMG_CSV_WAV_to_Spectrograms.ipynb**  
  This notebook takes the CSV files generated in the first preprocessing step and converts the labeled swallow events into WAV files. It then generates corresponding spectrograms for each event, which are used as input features for the machine learning models.

- **Principe Component Analysis.ipynb**  
  This notebook performs Principal Component Analysis (PCA) on the spectrogram data to reduce the dimensionality of the feature space. This step is for visualization purposes of the data distribution

- **SeMG_Spectrogram_Visualisation_.ipynb**  
  This notebook visualizes the spectrograms generated from the SeMG data. 

- **Testing_CNN_Segmenter_on_Test_dataset.ipynb**  
  This notebook tests the performance of the Convolutional Neural Network (CNN) segmenter on a test dataset. It evaluates the model's ability to accurately segment swallow events from the spectrogram data.

- **Training using MPS.ipynb**  
  This notebook trains the CNN models using Apple's Metal Performance Shaders (MPS) for accelerated training on Mac devices. It includes the implementation of the training loop and loss functions.

- **Training_Models_from_SeMG_Spectrograms (1).ipynb**  
  This notebook handles the training of various machine learning models on the SeMG spectrogram data. It includes the implementation of the CNN architecture, as well as experiments with different hyperparameters and model configurations.

## Key Components

### Spectrogram Generation
Spectrograms are generated from the WAV files created from SeMG data. These spectrograms serve as the primary input for the machine learning models. The frequency range and time window used in the spectrogram generation are critical for capturing the relevant features of the swallowing events.

### Machine Learning Models
The primary model used in this project is a Convolutional Neural Network (CNN) with a segmentation task to identify swallow events. Additionally, a regression task is implemented to estimate the percentage of the audio signal that corresponds to a swallow event.


## Future Work

- **Model Improvements**: Further optimization of the CNN architecture and hyperparameters to improve performance on the test dataset.
- **Feature Engineering**: Explore additional features or transformations of the spectrogram data to enhance model accuracy.
- **Real-time Application**: Develop a real-time application that can process and analyze SeMG data for swallow detection in real-time.
