from rlgames.common_types import Player, Point

COLS = 'ABCDEFGHJKLMNOPQRST'
STONE_TO_CHAR = {
  None : ' . ',
  Player.black : ' x ',
  Player.white : ' o ',
}

def print_move(player, move):
  if move.is_pass:
    move_str = 'passes'
  elif move.is_resign:
    move_str = 'resigns'
  else:
    move_str = '%s%d' % (COLS[move.pt.c - 1], move.pt.r)
  print('%s %s' % (player, move_str))

def print_board(board):
  for row in range(board.sz, 0, -1):
    bump = " " if row <= 9 else ""
    line = []
    for col in range(1, board.sz + 1):
      stone = board.get(Point(row, col))
      line.append(STONE_TO_CHAR[stone])
    print('%s%d %s' % (bump, row, ''.join(line)))
  print('    ' + '  '.join(COLS[:board.sz]))

def point_from_coord(coords):
  col = COLS.index(coords[0].upper()) + 1
  row = int(coords[1:])
  return Point(row, col)
