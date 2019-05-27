from common_types import Player
from game_base import Move
from goboard import GameState
from agents.random import RandomBot
from util import print_board, print_move, point_from_coord

def main():
  board_size = 9
  game = GameState.new_game(board_size)
  bot = RandomBot()
  while not game.is_over():
    print_board(game.board)
    if game.nplayer == Player.black:
      human_move = input('-- ')
      point = point_from_coord(human_move.strip())
      move = Move.play(point)
    else:
      move = bot.select_move(game)
    print_move(game.nplayer, move)
    game = game.apply_move(move)
  winner = game.winner()
  if winner is None:
    print("Tie")
  elif winner == Player.black:
    print("Black win")
  else:
    print("White win")
    
if __name__ == '__main__':
  main()
