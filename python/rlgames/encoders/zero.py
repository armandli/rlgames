import numpy as np

from rlgames.common_types import Point, Player
from rlgames.game_base import Move
from rlgames.goboard import GameState, GoString
from rlgames.encoders.base import Encoder

# AlphaGo Zero Encoder, does not use any convolution

# 0-3 are our stones with 1,2,3,4+ liberties
# 4-7 are opponent stones with 1,2,3,4+ liberties
# 8 1 if wen get komi
# 9 1 if opponent get komi
# 10 move would be illegal due to ko

class ZeroEncoder(Encoder):
  def __init__(self, sz):
    self.sz = sz
    self.num_planes = 11

  def encode(self, game_state):
    board_tensor = np.zeros(self.shape())
    base_plane = {game_state.nplayer: 0, game_state.nplayer.other : 3}
    next_player = game_state.nplayer
    if next_player == Player.white:
      board_tensor[8] = 1
    else:
      board_tensor[9] = 1
    for r in range(self.sz):
      for c in range(self.sz):
        p = Point(r + 1, c + 1)
        string = game_state.board.get_go_string_(p)
        if string is None:
          if game_state.does_move_violate_ko_(next_player, Move.play(p)):
            board_tensor[10][r][c] = 1
        else:
          liberty_plane = min(4, string.num_liberties) - 1
          liberty_plane += base_plane[string.color]
          board_tensor[liberty_plane][r][c] = 1
    return board_tensor

  def name(self):
    return "zero"

  # replacing decode point
  def encode_move(self, move):
    if move.is_play:
      return self.sz * move.pt.r + move.pt.c
    elif move.is_pass:
      return self.sz * self.sz
    raise ValueError('cannot decode resign move')

  #placeholder interface
  def encode_point(self, pt):
    return self.encode_move(Move.play(pt))

  # replacing decode_point_index
  def decode_move_index(self, index):
    if index == self.sz * self.sz:
      return Move.pass_turn()
    row = index // self.sz
    col = index % self.sz
    return Move.play(Point(row + 1, col + 1))

  #placeholder interface
  def decode_point_index(self, index):
    return self.decode_move_index(index)

  # replacing num_points
  def num_moves(self):
    return self.sz * self.sz + 1

  def shape(self):
    return self.num_planes, self.sz, self.sz

def create(board_size):
  return ZeroEncoder(board_size)
