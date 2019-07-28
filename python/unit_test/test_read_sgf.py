import argparse

from rlgames.goboard import GameState, Board
from rlgames.common_types import Point, Player
from rlgames.game_base import Move
from rlgames.sgf import sgf_game
from rlgames.util import print_board, print_move

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--file', '-f', type=str)
  args = parser.parse_args()
  return args

def new_game_from_handicap(sgf):
  board = Board(19)
  first_move_done = False
  move = None
  gs = GameState.new_game(19)
  if sgf.get_handicap() is not None and sgf.get_handicap() != 0:
    print('Handicap detected')
    for setup in sgf.get_root().get_setup_stones():
      for move in setup:
        row, col = move
        board.place_stone(Player.black, Point(row + 1, col + 1))
      first_move_done = True
      gs = GameState(board, Player.white, None, move)
  return gs, first_move_done

def main():
  args = parse_args()
  with open(args.file) as fd:
    data = fd.read()
    sgf = sgf_game.from_string(data)
    gs, first_move_done = new_game_from_handicap(sgf)
    print_board(gs.board)
    for item in sgf.main_sequence_iter():
      color, move_tuple = item.get_move()
      point = None
      if color is not None:
        if move_tuple is not None:
          row, col = move_tuple
          point = Point(row + 1, col + 1)
          move = Move.play(point)
          print('Move ({},{})'.format(row + 1, col + 1))
        else:
          move = Move.pass_turn()
        gs = gs.apply_move(move)
        print_board(gs.board)

if __name__ == '__main__':
  main()
