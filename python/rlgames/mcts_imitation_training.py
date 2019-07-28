import argparse
import numpy as np

from keras.layers.core import Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

from rlgames.mcts_models.convolutional import layers

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--trainx', '-x', type=str,
          default='/home/armandli/rlgames/data/mcts_features-40k.npy')
  parser.add_argument('--trainy', '-y', type=str,
          default='/home/armandli/rlgames/data/mcts_labels-40k.npy')
  parser.add_argument('--out', '-o', type=str,
          default='/home/armandli/rlgames/data/checkpoints/mcts_conv_model.h5')
  args = parser.parse_args()
  return args

def main():
  args = parse_args()
  np.random.seed(123)
  x = np.load(args.trainx)
  Y = np.load(args.trainy)
  samples = x.shape[0]
  size = 9
  input_shape = (size, size, 1)
  X = x.reshape(samples, size, size, 1)
  num_classes = size * size

  train_samples = int(0.9 * samples)
  X_train, X_test = X[:train_samples], X[train_samples:]
  Y_train, Y_test = Y[:train_samples], Y[train_samples:]

  # create keras convolutional model
  network_layers = layers(input_shape)
  model = Sequential()
  for layer in network_layers:
    model.add(layer)
  model.add(Dense(num_classes, activation='softmax'))
  model.summary()
  model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

  # training
  model.fit(X_train, Y_train, batch_size=64, epochs=100, verbose=1, validation_data=(X_test, Y_test))
  score = model.evaluate(X_test, Y_test, verbose=0)
  print('Test Loss:', score[0])
  print('Test Accuracy:', score[1])

  # saving result
  model_file = args.out
  model.save(model_file,overwrite=True,include_optimizer=False)

if __name__ == '__main__':
  main()
