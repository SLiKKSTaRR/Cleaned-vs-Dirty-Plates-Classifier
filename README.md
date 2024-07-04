# Cleaned vs Dirty Plates Classifier

This project is a deep learning model for classifying images of plates as either clean or dirty using transfer learning with the MobileNetV2 architecture. The dataset used for training and testing is part of a Kaggle competition.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Prediction](#prediction)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This repository contains code to train a convolutional neural network (CNN) using transfer learning to classify images of plates as clean or dirty. The MobileNetV2 model, pre-trained on ImageNet, is used as the base model to leverage the features learned from a large dataset.

## Dataset

The dataset is a zip file containing images of plates divided into two categories: 'cleaned' and 'dirty'. The dataset is extracted from a Kaggle competition.

## Installation

To use this repository, follow these steps:

1. Clone this repository:
    ```sh
    git clone https://github.com/slikkstarr/cleaned-vs-dirty-plates.git
    cd cleaned-vs-dirty-plates
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Extract Dataset**: Ensure the dataset zip file is located in the `input/` directory. Extract the dataset:

2. **Define Paths**: Set the paths to the train and test directories:


## Training
- Train the model using the provided script


## Prediction
- Use the trained model to make predictions on the test set


## Results
- The results will be saved in a CSV file named submission.csv in the working directory. The CSV file contains the IDs of the test images and their predicted labels


## Contributing
- Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
- This project is licensed under the MIT License. See the LICENSE file for details.
