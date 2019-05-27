from common_types import Point

# this is a flawed definition of an eye
def is_point_an_eye(board, point, color):
  # an eye is an empty point
  if board.get(point) is not None:
    return False
  # all adj points must be friendly stones
  for neighbour in point.neighbours():
    if board.is_on_grid(neighbour):
      neighbour_color = board.get(neighbour)
      if neighbour_color != color:
        return False
  # must control 3 out of 4 corners if the point is in the middle of the board,
  # on edge it must control all corners
  friendly_corners = 0
  off_board_corners = 0
  for corner in point.corners():
    if board.is_on_grid(corner):
      corner_color = board.get(corner)
      if corner_color == color:
        friendly_corners += 1
    else:
      off_board_corners += 1
  if off_board_corners > 0:
    return off_board_corners + friendly_corners == 4
  return friendly_corners >= 3
