import time
from common_types import Player
from tictactoe_board import GameState
from agents.minimax import MinimaxAgent
from util import print_board, print_move

def main():
  board_size = 3
  game = GameState.new_game(board_size)
  bots = {
    Player.black : MinimaxAgent(5, None),
    Player.white : MinimaxAgent(5, None),
  }
  while not game.is_over():
    time.sleep(0.1)

    print_board(game.board)
    bot_move = bots[game.nplayer].select_move(game)
    print_move(game.nplayer, bot_move)
    game = game.apply_move(bot_move)
  winner = game.winner()
  if winner is None:
    print("Tie")
  elif winner == Player.black:
    print("Black won")
  else:
    print("White won")

if __name__ == '__main__':
  main()
