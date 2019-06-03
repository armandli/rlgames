from abc import abstractmethod

# There are 3 types of moves player can make: place piece, pass, or resign
# a good bot should know how to resign
class Move:
  def __init__(self, point=None, is_pass=False, is_resign=False):
    assert (point is not None) ^ is_pass ^ is_resign
    self.pt = point
    self.is_play = self.pt is not None
    self.is_pass = is_pass
    self.is_resign = is_resign

  @classmethod
  def play(cls, point):
    return cls(point)

  @classmethod
  def pass_turn(cls):
    return cls(is_pass = True)

  @classmethod
  def resign(cls):
    return cls(is_resign = True)

class BoardBase(object):
  @abstractmethod
  def is_on_grid(self, pt):
    raise NotImplementedError()
  @abstractmethod
  def get(self, pt):
    raise NotImplementedError()
  @abstractmethod
  def place_stone(self, player, pt):
    raise NotImplementedError()

class GameStateBase(object):
  @abstractmethod
  def apply_move(self, move):
    raise NotImplementedError()
  @abstractmethod
  def is_over(self):
    raise NotImplementedError()
  @abstractmethod
  def is_valid_move(self, move):
    raise NotImplementedError()
  @abstractmethod
  def legal_moves(self):
    raise NotImplementedError()
  @abstractmethod
  def winner(self):
    raise NotImplementedError()
