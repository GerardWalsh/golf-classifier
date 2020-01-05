from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.applications import vgg16, resnet
from tensorflow.keras.optimizers import Adagrad, SGD
from tensorflow.keras.callbacks import LearningRateScheduler

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.utils.model import build_model
from src.utils.data import get_images_path
from src.utils.learning_rate_schedulers import PolynomialDecay

# Constants
BATCH_SIZE = 32
REPLICATES = 4
IMG_WIDTH = 224
IMG_HEIGHT = 224
EPOCHS = 85
INIT_LR = 5e-3
CHANNELS = 3

# Prepare data
print('[INFO] Getting dataset from disk....')
train_dir = 'car-dataset/data'
images, labels = get_images_path('car-dataset/data')
classes = 0

for label, count in zip(Counter(labels).keys(), Counter(labels).values()):
    print('Label {} has count {}.'.format(label, count))
    classes+=1

CLASSES = classes

print('[INFO] Performing data augementation....')
train_image_generator = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.25,
    horizontal_flip=True,
    fill_mode='nearest', 
)

def augment_image(image, replicates=4):
    images = [image]
    i = 1
    for batch in train_image_generator.flow(image, batch_size=1):
        images.append(batch)
        i += 1
        if i >= replicates:
            break
    return images

# Augment each image and produce replicates, as a list, and subsequently transform 2D list into 1D list
images = [augment_image(image, REPLICATES) for image in images]
images = [image for sublist in images for image in sublist]

# Create copies of each label, IN ORDER, and transform from 2D to 1D
labez = [[label] * REPLICATES for label in labels]
labelz = np.array([label for sublist in labez for label in sublist])

images = np.concatenate(images)
input_shape = images.shape[1:]

# Preprocess images according ResNet procedure (ImageNet procedure)
# print('[INFO] Preprocessing images....')
images = resnet.preprocess_input(images)
(X_train, X_test, y_train, y_test) = train_test_split(images, labelz, test_size=0.02, random_state=13, shuffle=True)

# One-hot encode labels
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()

y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

# Define model
print('[INFO] Building model....')
base_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
vgg_ = build_model(base_model, classes=CLASSES, base_trainable=True)

# Define loss and optimizer
loss = CategoricalCrossentropy(label_smoothing=0.1)
opt = SGD(lr=INIT_LR, momentum=0.9)

# Compile model
vgg_.compile(
    loss=loss,
    optimizer=opt,
    metrics=['accuracy']
)

# Define callback to save model weights
checkpoint_path = "results/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    save_weights_only=True, 
    verbose=1)

# construct the learning rate scheduler callback
schedule = PolynomialDecay(maxEpochs=EPOCHS, initAlpha=INIT_LR,
	power=1.0)
callbacks = [LearningRateScheduler(schedule), cp_callback]


# Train model
history = vgg_.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1,
    validation_split=0.2,
    callbacks=callbacks
)

# Report
predictions = vgg_.predict(X_test, batch_size=BATCH_SIZE)
print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1)))