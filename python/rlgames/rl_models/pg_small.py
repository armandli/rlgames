from keras.models import Model
from keras.layers import Conv2D, Dense, Flatten, Input, ZeroPadding2D

from rlgames.encoders.base import Encoder

def pg_model(encoder):
  board_input = Input(encoder.shape(), name='board_input')
  padi1 = ZeroPadding2D(padding=3, input_shape=encoder.shape(), data_format='channels_first')(board_input)
  conv1 = Conv2D(48, (7, 7), data_format='channels_first', activation='relu')(padi1)

  conv2 = Conv2D(32, (5, 5), padding='same', activation='relu')(conv1)
  conv3 = Conv2D(32, (5, 5), padding='same', activation='relu')(conv2)
  conv4 = Conv2D(32, (5, 5), padding='same', activation='relu')(conv3)

  flat1 = Flatten()(conv4)
  dens1 = Dense(512, activation='relu')(flat1)
  dens2 = Dense(encoder.num_points(), activation='softmax')(dens1)

  model = Model(inputs=board_input, outputs=dens2)
  return model
