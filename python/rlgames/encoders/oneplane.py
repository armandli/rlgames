import numpy as np

from rlgames.encoders.base import Encoder
from rlgames.common_types import Point

class OnePlaneEncoder(Encoder):
  def __init__(self, board_size):
    self.sz = board_size
    self.num_planes = 1

  def name(self):
    return 'oneplane'

  def encode(self, game_state):
    board_mtx = np.zeros(self.shape())
    nplayer = game_state.nplayer
    for r in range(self.sz):
      for c in range(self.sz):
        p = Point(r+1, c+1)
        gstring = game_state.board.get_go_string_(p)
        if gstring is None:
          continue
        if gstring.color == nplayer:
          board_mtx[0, r, c] = 1
        else:
          board_mtx[0, r, c] = -1
    return board_mtx

  def encode_point(self, pt):
    return self.sz * (pt.r - 1) + (pt.c - 1)

  def decode_point_index(self, index):
    row = index // self.sz
    col = index % self.sz
    return Point(row + 1, col + 1)

  def num_points(self):
    return self.sz * self.sz

  def shape(self):
    return self.num_planes, self.sz, self.sz

def create(board_size):
  return OnePlaneEncoder(board_size)
