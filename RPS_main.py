import os
from keras import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.src.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import cv2
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

cameraNum = 0
testRatio = 0.15
ValidationRation = 0.15
path = 'C:/Users/Nour/Desktop/Github/RockPaperScissors/Dataset'
class_names = ['paper', 'rock', 'scissors']
imgDim = (300, 300, 3)

images = []
classNo = []

print(path, "contains:", class_names)
numOfClasses = len(class_names)
print("Importing data...")
for classIndex, className in enumerate(class_names):
    imageList = os.listdir(os.path.join(path, className))
    for imageFile in imageList:
        curImg = cv2.imread(os.path.join(path, className, imageFile))
        curImg = cv2.resize(curImg, (50, 50))
        images.append(curImg)
        classNo.append(classIndex)
    print(className, end=".")

print("")
images = np.array(images)
classNo = np.array(classNo)
print(images.shape)
print(classNo.shape)


x_train, x_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=ValidationRation)
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


x_train = np.array(list(map(preprocess, x_train)))
x_test = np.array(list(map(preprocess, x_test)))
x_validation = np.array(list(map(preprocess, x_validation)))

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[1], 1)
x_validation = x_validation.reshape(x_validation.shape[0], x_validation.shape[1], x_validation.shape[1], 1)

print("x train : ", x_train.shape)
print("x test :  ", x_test.shape)
numOfSamples = []
for x in range(0, numOfClasses):
    numOfSamples.append(len(np.where(y_train == x)[0]))
print(numOfSamples)


plt.figure(figsize=(8, 4))
plt.bar(range(0, numOfClasses), numOfSamples)
plt.title("Nombre of images for each class")
plt.xlabel("class ID")
plt.ylabel("nombre d'images")
plt.show()

random_index = np.random.randint(0, x_train.shape[0])
random_image = x_train[random_index]

plt.figure(figsize=(2,2))
plt.imshow(random_image.squeeze(), cmap='gray')
plt.title(f"Random Image from Dataset at Index {random_index}")
plt.axis('off')  # Hide axes ticks
plt.show()

# the data augmentation is skipped
y_train = to_categorical(y_train, len(class_names))
y_test = to_categorical(y_test, len(class_names))
y_validation = to_categorical(y_validation, len(class_names))

noOfFilters = 60
sizeOfFilter1 = (5, 5)
sizeOfFilter2 = (3, 3)
sizeOfPool = (2, 2)
noOfNodes = 500
imgDim = (50, 50, 1)
batchSize = 60
epochVal = 5
def myModel():
    model = Sequential()
    model.add(Conv2D(noOfFilters, sizeOfFilter1, input_shape=(imgDim[0], imgDim[1], 1), activation='relu'))
    model.add(Conv2D(noOfFilters, sizeOfFilter1, activation='relu'))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu'))
    model.add(Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu'))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(noOfNodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(class_names), activation='softmax'))
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = myModel()
print(model.summary())

history = model.fit(x_train, y_train, batch_size=batchSize, steps_per_epoch=len(x_train) // batchSize, epochs=epochVal, validation_data=(x_validation, y_validation), shuffle=1)

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
plt.show()
plt.show()
score = model.evaluate(x_test, y_test, verbose=0)
print("test Score = ", score[0])
print("test Acc = ", score[1])


tf.saved_model.save(
    model, "C:/Users/Nour/Desktop/Github/RockPaperScissors/RPS.tf")


