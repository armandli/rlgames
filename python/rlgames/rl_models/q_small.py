from keras.models import Model
from keras.layers import Conv2D, Dense, Flatten, Input, ZeroPadding2D, concatenate

from rlgames.encoders.base import Encoder

#qlearning model uses CNN for the board state encoding, but need to append
#action into the neural network used for function approximiation
def qlearning_model(encoder):
  #required to use functional style network description due to the need to
  #concatenate state and action input matrix
  board_input = Input(shape=encoder.shape(), name='board_input')
  action_input = Input(shape=(encoder.num_points(),), name='action_input')
  conv1a = ZeroPadding2D((2, 2))(board_input)
  conv1b = Conv2D(64, (5, 5), activation='relu')(conv1a)
  conv2a = ZeroPadding2D((1, 1))(conv1b)
  conv2b = Conv2D(64, (3, 3), activation='relu')(conv2a)
  flat = Flatten()(conv2b)
  process_board = Dense(512)(flat)
  board_and_action = concatenate([action_input, process_board])
  hidden_layer = Dense(256, activation='relu')(board_and_action)
  value_output = Dense(1, activation='tanh')(hidden_layer)
  model = Model(input=[board_input, action_input], output=value_output)
  return model
