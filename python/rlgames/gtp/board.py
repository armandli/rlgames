from rlgames.common_types import Point
from rlgames.game_base import Move

COLS = 'ABCDEFGHJKLMNOPQRST'

def coord_to_gtp_position(move):
  point = move.pt
  return COLS[point.c - 1] + str(point.r)

def gtp_position_to_coord(gtp_position):
  col_str, row_str = gtp_position[0], gtp_position[1:]
  point = Point(int(row_str), COLS.find(col_str.upper()) + 1)
  return Move.play(point)
