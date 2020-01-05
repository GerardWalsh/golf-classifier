import os
import numpy as np
from keras_preprocessing.image import load_img
from keras_preprocessing.image import ImageDataGenerator
from imutils import paths
import cv2
import tensorflow as tf
import pathlib

REPLICATES = 4

def get_images_path(image_path):

    image_paths = list(paths.list_images(image_path))

    images = []
    labels = []

    for path in image_paths:
        img = load_img(path, target_size=(224, 224))
        img = np.expand_dims(img, axis=0)
        
        images.append(img)

        data_point_label = path.split(os.path.sep)[-2]

        labels.append(data_point_label)

    return images, labels


def augment_image(image, datagen, replicates=4):
    images = [image]
    i = 1 
    for batch in datagen.flow(image, batch_size=1):
        images.append(batch)
        i += 1
        if i >= replicates:
            break
    return images


def augment_dataset(input_images, labels, **kwargs):

    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.25,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    images = [augment_image(image, datagen, REPLICATES) for image in input_images] 

    images = [image for sublist in images for image in sublist]

    labez = [[label] * REPLICATES for label in labels]
    labelz = np.array([label for sublist in labez for label in sublist])

    images = np.concatenate(images)

    input_shape = images.shape[1:]

    images = resnet.preprocess_input(images)

    return images, labelz, input_shape

