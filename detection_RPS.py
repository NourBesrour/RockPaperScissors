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

    # Make predictionqqq
    predictions = serving_signature(conv2d_input=input_tensor)

    # Extract the prediction result
    probval = np.max(predictions['dense_1'].numpy())
    classIndex = np.argmax(predictions['dense_1'].numpy(), axis=1)

    class_names = ['paper', 'rock', 'scissors']
    if probval > limite:
        class_name = class_names[classIndex[0]]
        print(
            f"Detected object: {class_name}, Probability: {probval * 100:.2f}%")
        cv2.putText(sharpened, f'{class_name} {probval*100:.2f}%',
                    (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Processed", sharpened)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
