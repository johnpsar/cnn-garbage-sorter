from tensorflow import keras

import numpy as np
import tensorflow as tf
from tensorflow import keras

import pygame
import pygame.camera

pygame.camera.init()
pygame.camera.list_cameras()  # Camera detected or not
cam = pygame.camera.Camera("/dev/video0", (500, 500)) #change this to your video device
cam.start()
img = cam.get_image()
img = cam.get_image()

pygame.image.save(img, "./capture.jpg") #save image as capture.jpg

image_size = (500, 500)

model = keras.models.load_model('./trained_model_12.h5') #load trained model

img = keras.preprocessing.image.load_img(
    "./capture.jpg", target_size=image_size
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
score = predictions[0]
print(
    "This  photo is %.2f percent banana and %.2f percent hell."
    % (100 * (1 - score), 100 * score)
)

