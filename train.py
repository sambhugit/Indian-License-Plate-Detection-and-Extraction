import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Conv2D
from tensorflow.keras import optimizers
import datetime

current_dir = os.path.dirname(os.path.realpath('inference.py'))

train_datagen = ImageDataGenerator(rescale=1./255, width_shift_range=0.1, height_shift_range=0.1)
train_generator = train_datagen.flow_from_directory(
        'data/train',  # this is the target directory
        target_size=(28,28),  # all images will be resized to 28x28
        batch_size=1,
        class_mode='categorical')

validation_generator = train_datagen.flow_from_directory(
        'data/val',  # this is the target directory
        target_size=(28,28),  # all images will be resized to 28x28        batch_size=1,
        class_mode='categorical')

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

def model_f():

    model = Sequential()
    model.add(Conv2D(32, (24,24), input_shape=(28, 28, 3), activation='relu', padding='same'))
    # model.add(Conv2D(32, (20,20), input_shape=(28, 28, 3), activation='relu', padding='same'))
    # model.add(Conv2D(32, (20,20), input_shape=(28, 28, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(36, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.00001), metrics=['accuracy'])

    return model

class stop_training_callback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
      if logs.get('val_acc') is not None and logs.get('val_acc') > 0.950:
          self.model.stop_training = True

if __name__ == "__main__":

    model = model_f()

    batch_size = 1
    callbacks = [tensorboard_callback, stop_training_callback()]
    model.fit_generator(
          train_generator,
          steps_per_epoch = train_generator.samples // batch_size,
          validation_data = validation_generator, 
          validation_steps = validation_generator.samples // batch_size,
          epochs = 80, callbacks=callbacks)
    model.save(os.path.join(current_dir,"Model/License_plate.hdf5"))
