import numpy as np

from rlgames.encoders.base import Encoder
from rlgames.common_types import Point
from rlgames.game_base import Move

# Seven Plane Encoder: planes 0-2 encode white go string with 1,2,3+ liberties respectively,
# planes 3-5 encode black go string with 1,2,3+ liberties respectively, plane 6 encodes ko 
# violation points

class SevenPlaneEncoder(Encoder):
  def __init__(self, board_size):
    self.sz = board_size
    self.num_planes = 7

  def name(self):
    return 'sevenplane'

  def encode(self, game_state):
    board_tensor = np.zeros(self.shape())
    base_plane = {game_state.nplayer: 0, game_state.nplayer.other : 3}
    for row in range(self.sz):
      for col in range(self.sz):
        p = Point(r = row + 1, c = col + 1)
        gostring = game_state.board.get_go_string_(p)
        if gostring is None:
          if game_state.does_move_violate_ko_(game_state.nplayer, Move.play(p)):
            board_tensor[6][row][col] = 1
        else:
          liberty_plane = min(3, gostring.num_liberties) - 1
          liberty_plane += base_plane[gostring.color]
          board_tensor[liberty_plane][row][col] = 1
    return board_tensor

  def encode_point(self, pt):
    return self.sz * (pt.r - 1) + (pt.c - 1)

  def decode_point_index(self, index):
    row = index // self.sz
    col = index % self.sz
    return Point(r=row + 1, c=col + 1)

  def num_points(self):
    return self.sz * self.sz

  def shape(self):
    return self.num_planes, self.sz, self.sz

def create(board_size):
  return SevenPlaneEncoder(board_size)
