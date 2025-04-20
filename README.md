# Face Recognition with CNN and Eigenfaces

  <p align="center">
    <p align="center">
    <a href="https://github.com/LoanClt/FacialRecognition">See a demo</a>
    ·
    <a href="https://github.com/LoanClt">Report a bug</a>
    ·
    <a href="https://github.com/LoanClt">Ask for a feature</a>
  </p>

This repository contains Python scripts for face recognition using two different approaches:
1. **Convolutional Neural Networks (CNN)**: A deep learning approach for facial recognition.
2. **Eigenfaces with PCA and k-NN**: A traditional machine learning approach using Principal Component Analysis (PCA) and k-Nearest Neighbors (k-NN).

## List of files 

1. `augmentedDataSet.py`: Artificialy improve the size of a given dataset
2. `cnn.py`: Face recognition using a CNN (2 Conv2D + MaxPooling and a fully connected layer)
3. `eigenfaces.py`: Face recognition using eigenfaces method
4. `imageEditor.pr`: Compress and lower the quality of a dataset

## Features
- **Image Preprocessing**: Grayscale conversion, resizing, and normalization.
- **Data Augmentation**: Increases dataset size using transformations like flipping, brightness adjustment, noise addition, rotation, and blurring.
- **CNN Model**: Built using TensorFlow/Keras with convolutional and dense layers.
- **Eigenface Recognizer**: Uses PCA for dimensionality reduction and k-NN for classification.
- **Evaluation Metrics**: Includes accuracy calculation, confusion matrix, and visualization of predictions.
- **Visualization Tools**: Displays eigenfaces, test results, and model performance.

## Installation
```sh
pip install -r requirements.txt
```

## Usage
## CNN Model
Run the script to train a CNN model on the dataset:
```sh
python cnn.py
```

## Eigenface Recognizer
Run the script to train a CNN model on the dataset:
```sh
python eigenfaces.py
```

## Dataset Structure
```sh
Dataset/
    ├── subject1/
    │   ├── img1.jpg
    │   ├── img2.jpg
    ├── subject2/
    │   ├── img1.jpg
    │   ├── img2.jpg
```




Matthew Turk, Alex Pentland; Eigenfaces for Recognition. J Cogn Neurosci 1991; 3 (1): 71–86. doi: https://doi.org/10.1162/jocn.1991.3.1.71


## Authors

- KINA Matheo and Loan CHALLEAT


## Running Tests

Tests run on the famous Yale dataset : https://www.kaggle.com/datasets/preprocessiing/cropped-faces
Recognition accuracy using eigenFaces method: **92.58%**
Using a CNN : **99.2%**

Here are the results using the eigenface smethod:

<img src="/img/result.png">
<img src="/img/confusionMatrix.png">
<img src="/img/eigenFaces.png">


## Feedback

If you have any feedback, please reach out to us at loan.challeat@institutoptique.fr

