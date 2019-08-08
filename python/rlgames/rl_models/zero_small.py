from keras.models import Model
from keras.layers import Conv2D, Dense, Flatten, Input

from rlgames.encoders.base import Encoder

# example alpha go zero model

def zero_model(encoder):
  board_input = Input(encoder.shape(), name='board_input')
  prev_layer = board_input
  for i in range(4):
    prev_layer = Conv2D(64, (3, 3), padding='same', activation='relu')(prev_layer)
  policy_conv = Conv2D(2, (1, 1), data_format='channels_first', activation='relu')(prev_layer)
  policy_flat = Flatten()(policy_conv)
  policy_out  = Dense(encoder.num_moves(), activation='softmax')(policy_flat)
  value_conv  = Conv2D(1, (1, 1), data_format='channels_first', activation='relu')(prev_layer)
  value_flat  = Flatten()(value_conv)
  value_hid   = Dense(256, activation='relu')(value_flat)
  value_out   = Dense(1, activation='tanh')(value_hid)
  model = Model(inputs=board_input, outputs=[policy_out, value_out])
  return model
