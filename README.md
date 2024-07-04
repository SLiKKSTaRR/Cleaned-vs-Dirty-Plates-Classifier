# Cleaned vs Dirty Plates Classifier
This project is a deep learning model for classifying images of plates as either clean or dirty using transfer learning with the MobileNetV2 architecture. The dataset used for training and testing is part of a Kaggle competition.

Table of Contents
Introduction
Dataset
Installation
Usage
Training
Prediction
Results
Contributing
License
Introduction
This repository contains code to train a convolutional neural network (CNN) using transfer learning to classify images of plates as clean or dirty. The MobileNetV2 model, pre-trained on ImageNet, is used as the base model to leverage the features learned from a large dataset.

Dataset
The dataset is a zip file containing images of plates divided into two categories: 'cleaned' and 'dirty'. The dataset is extracted from a Kaggle competition.

Installation
To use this repository, follow these steps:

Clone this repository:

sh
Copy code
git clone https://github.com/yourusername/cleaned-vs-dirty-plates.git
cd cleaned-vs-dirty-plates
Install the required packages:

sh
Copy code
pip install -r requirements.txt
Usage
Extract Dataset: Ensure the dataset zip file is located in the input/ directory. Extract the dataset:

python
Copy code
import os
import zipfile

dataset_path = '/kaggle/input/platesv2/plates.zip'
extract_dir = '/kaggle/working/'

with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print("Dataset extracted successfully to:", extract_dir)
Define Paths: Set the paths to the train and test directories:

python
Copy code
train_dir = '/kaggle/working/plates/train/'
test_dir = '/kaggle/working/plates/test/'
Training
Train the model using the provided script:

python
Copy code
import numpy as np
import pandas as pd
import os
import zipfile
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Define data generators for augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 32
)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
Prediction
Use the trained model to make predictions on the test set:

python
Copy code
def load_and_preprocess_test_images(test_dir, target_size=(224, 224)):
    test_images = []
    test_ids = []

    for filename in os.listdir(test_dir):
        img_path = os.path.join(test_dir, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, target_size)
            img = preprocess_input(img)
            test_images.append(img)
            test_ids.append(filename.split('.')[0])

    test_images = np.array(test_images)
    return test_images, test_ids

test_images, test_ids = load_and_preprocess_test_images(test_dir)
predictions = model.predict(test_images)
threshold = 0.5
predicted_labels = ['cleaned' if pred > threshold else 'dirty' for pred in predictions]

submission_df = pd.DataFrame({'id': test_ids, 'label': predicted_labels})
submission_df.to_csv('/kaggle/working/submission.csv', index=False)
print("Submission file saved to /kaggle/working/submission.csv")
Results
The results will be saved in a CSV file named submission.csv in the working directory. The CSV file contains the IDs of the test images and their predicted labels.

Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

License
This project is licensed under the MIT License. See the LICENSE file for details.

