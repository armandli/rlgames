import importlib
from abc import abstractmethod

def get_encoder_by_name(name, board_size):
  module = importlib.import_module('rlgames.encoders.' + name)
  constructor = getattr(module, 'create')
  return constructor(board_size)

class Encoder:
  @abstractmethod
  def name(self):
    """
    name of the encoder
    """
    raise NotImplementedError()
  @abstractmethod
  def encode(self, game_state):
    """
    encode a game state into a vector of features
    """
    raise NotImplementedError()
  @abstractmethod
  def encode_point(self, point):
    """
    turns a go board point into a index
    """
    raise NotImplementedError()
  @abstractmethod
  def decode_point_index(self, index):
    """
    turns index back into go board point
    """
    raise NotImplementedError()
  @abstractmethod
  def num_points(self):
    """
    number of points on the board
    """
    raise NotImplementedError()
  @abstractmethod
  def shape(self):
    """
    shape of the encoded board structure
    """
    raise NotImplementedError()
