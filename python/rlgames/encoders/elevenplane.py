import numpy as np

from rlgames.encoders.base import Encoder
from rlgames.common_types import Point, Player
from rlgames.game_base import Move

# Eleven Plane Encoder: planes 0-3 encode white go string with 1,2,3,4+ liberties respectively,
# planes 4-7 encode black go string with 1,2,3,4+ liberties respectively, plane 8,9 encodes whether
# it is black or white's turn, plane 10 encodes ko violation points

class ElevenPlaneEncoder(Encoder):
  def __init__(self, board_size):
    self.sz = board_size
    self.num_planes = 11

  def name(self):
    return 'elevenplane'

  def encode(self, game_state):
    board_tensor = np.zeros(self.shape())
    base_plane = {Player.white: 4, Player.black: 0}
    if game_state.nplayer == Player.black:
      board_tensor[8] = 1
    else:
      board_tensor[9] = 1
    for row in range(self.sz):
      for col in range(self.sz):
        p = Point(r = row + 1, c = col + 1)
        gostring = game_state.board.get_go_string_(p)
        if gostring is None:
          if game_state.does_move_violate_ko_(game_state.nplayer, Move.play(p)):
            board_tensor[10][row][col] = 1
        else:
          liberty_plane = min(4, gostring.num_liberties) - 1
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
  return ElevenPlaneEncoder(board_size)
