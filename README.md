
# Rock, Paper, Scissors Classification

This project implements a Convolutional Neural Network (CNN) to classify images of Rock, Paper, and Scissors using TensorFlow and Keras. The model is trained on a custom dataset and can make predictions based on real-time video input from a camera.

## Table of Contents
- [Project Description](#project-description)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Results](#results)
- [License](#license)
- [Video Demonstration](#video-demonstration)

## Project Description

The goal of this project is to develop a machine learning model capable of accurately classifying images into one of three classes: Rock, Paper, or Scissors. The model uses a CNN architecture and is trained on images collected from a dataset.

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/YourUsername/RockPaperScissors.git
   cd RockPaperScissors
   ```

2. **Install Required Packages**
   Ensure you have Python 3.x installed, then install the necessary libraries:
   ```bash
   pip install tensorflow opencv-python matplotlib numpy
   ```

3. **Prepare the Dataset**
   - Organize your dataset in the following structure:
     ```
     Dataset/
         paper/
             image1.jpg
             image2.jpg
             ...
         rock/
             image1.jpg
             image2.jpg
             ...
         scissors/
             image1.jpg
             image2.jpg
             ...
     ```

## Usage

1. **Train the Model**
   - Run the `RPS_main.py` script to train the model.
   ```bash
   python RPS_main.py
   ```

2. **Run the Prediction Script**
   - Use the `RPS_prediction.py` script to make predictions based on video input.
   ```bash
   python RPS_prediction.py
   ```

## Data Preparation

- Images are resized to 50x50 pixels and converted to grayscale.
- The dataset is split into training, validation, and testing sets with a 70-15-15 ratio.
- Data is preprocessed using histogram equalization for better contrast.

## Model Architecture

The CNN model architecture consists of the following layers:
- **Convolutional Layers**: 
  - 2 layers of Conv2D with 60 filters and a kernel size of 5x5, followed by MaxPooling.
  - 2 layers of Conv2D with 30 filters and a kernel size of 3x3, followed by MaxPooling.
- **Dropout Layers**: Added after the pooling layers to prevent overfitting.
- **Dense Layers**: 
  - Flatten layer followed by a Dense layer with 500 nodes and a final Dense layer with 3 output nodes (for Rock, Paper, Scissors).

### Summary of Model

```python
from keras import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.src.optimizers import Adam

def myModel():
    model = Sequential()
    model.add(Conv2D(60, (5, 5), input_shape=(50, 50, 1), activation='relu'))
    model.add(Conv2D(60, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = myModel()
print(model.summary())
```

## Training Process

- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Epochs**: 5
- **Batch Size**: 60

### Training Results
Training and validation loss and accuracy are plotted for evaluation.

```python
history = model.fit(x_train, y_train, batch_size=60, steps_per_epoch=len(x_train) // 60, epochs=5, validation_data=(x_validation, y_validation), shuffle=1)

import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('Epoch')
```

## License

This project is licensed under the MIT License.

## Image Resizing Script

Included is a script to resize images in the dataset to 50x50 pixels.

```python
import os
from PIL import Image

# Path to the GTSRB folder
base_folder = 'C:/Users/Nour/Desktop/Github/GTSRB/GTSRB/Train'
resize_folder = os.path.join(base_folder, 'resize')

# Create the resize folder if it doesn't exist
os.makedirs(resize_folder, exist_ok=True)

# Loop through each subfolder in the train folder
for label_folder in os.listdir(base_folder):
    label_folder_path = os.path.join(base_folder, label_folder)
    
    # Check if it's a directory
    if os.path.isdir(label_folder_path):
        # Create a corresponding folder in the resize directory
        resized_label_folder = os.path.join(resize_folder, label_folder)
        os.makedirs(resized_label_folder, exist_ok=True)
        
        # Loop through each image in the label folder
        for image_name in os.listdir(label_folder_path):
            image_path = os.path.join(label_folder_path, image_name)
            if image_path.endswith(('jpg', 'jpeg', 'png')):  # Check for image files
                # Open and resize the image
                with Image.open(image_path) as img:
                    img = img.convert('L')  # Convert to grayscale
                    img = img.resize((50, 50))  # Resize to 50x50
                    # Save the resized image
                    resized_image_path = os.path.join(resized_label_folder, image_name)
                    img.save(resized_image_path)

print("Resizing complete!")
```

## Real-time Prediction Script

A script is provided to use the trained model for real-time predictions from the webcam.

```python
import tensorflow as tf
import cv2
import numpy as np
from RPS_main import preprocess

cameraNum = 0
limite = 0.7

# Load the model using TensorFlow SavedModel API
modelPath = "C:/Users/Nour/Desktop/Github/RockPaperScissors/RPS.tf"
loaded_model = tf.saved_model.load(modelPath)

# Get the serving signature for making predictions
serving_signature = loaded_model.signatures['serving_default']

cap = cv2.VideoCapture(cameraNum)
cap.set(15, 15)
cap.set(3, 640)
cap.set(4, 480)

while True:
    success, imgorg = cap.read()
    if not success:
        break

    imgorg = cv2.GaussianBlur(imgorg, (5, 5), 0)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(imgorg, -1, kernel)
    img = np.asarray(sharpened)
    img = cv2.resize(img, (50, 50))
    img1 = preprocess(img)
    imgp = img1.reshape(1, 50, 50, 1)

    input_tensor = tf.constant(imgp, dtype=tf.float32)

    # Make prediction
    predictions = serving_signature(conv2d_input=input_tensor)

    # Extract the prediction result
    probval = np.max(predictions['dense_1'].numpy())
    classIndex = np.argmax(predictions['dense_1'].numpy(), axis=1)

    class_names = ['paper', 'rock', 'scissors']
    if probval > limite:
        class_name = class_names[classIndex[0]]
        print(f"Detected object: {class_name}, Probability: {probval * 100:.2f}%")
        cv2.putText(sharpened, f'{class_name} {probval*100:.2f}%', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Processed", sharpened)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Video Demonstration

You can view a demonstration of the project in action here:

<iframe src="https://www.linkedin.com/embed/feed/update/urn:li:ugcPost:7178022260379697152" height="730" width="504" frameborder="0" allowfullscreen="" title="Post intégré"></iframe>




