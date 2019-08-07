from keras.models import Model
from keras.layers import Conv2D, Dense, Flatten, Input

from rlgames.encoders.base import Encoder

#small actor critic model

def actor_critic_model(encoder):
  board_input = Input(encoder.shape(), name='board_input')
  conv1 = Conv2D(64, (3, 3), padding='same', activation='relu')(board_input)
  conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv1)
  conv3 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv2)
  flat  = Flatten()(conv3)
  processed_board = Dense(512)(flat)
  policy_hidden_layer = Dense(512, activation='relu')(processed_board)
  policy_output = Dense(encoder.num_points(), activation='softmax')(policy_hidden_layer)
  value_hidden_layer = Dense(512, activation='relu')(processed_board)
  value_output = Dense(1, activation='tanh')(value_hidden_layer)
  model = Model(inputs=board_input, outputs=[policy_output, value_output])
  return model
