from keras.layers.core import  Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D

def layers(input_shape):
  return [
    Conv2D(48, kernel_size=(3,3), activation='relu', padding='same',
        input_shape=input_shape),
    Dropout(rate=0.5),
    Conv2D(48, (3,3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(rate=0.5),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(rate=0.5)
  ]
