import time
from rlgames.common_types import Player
from rlgames.goboard import GameState
from rlgames.agents.random import RandomAgent
from rlgames.agents.random_fast import FastRandomAgent
from rlgames.util import print_board, print_move

def main():
  board_size = 9
  game = GameState.new_game(board_size)
  bots = {
#    Player.black : RandomAgent(),
#    Player.white : RandomAgent(),

    Player.black : FastRandomAgent(board_size),
    Player.white : FastRandomAgent(board_size),
  }
  while not game.is_over():
    time.sleep(0.1) # slow down so we can observe

    #print(chr(27) + "[2J") #clear screen
    print_board(game.board)
    bot_move = bots[game.nplayer].select_move(game)
    print_move(game.nplayer, bot_move)
    game = game.apply_move(bot_move)
  winner = game.winner()
  if winner is None:
    print("Tie")
  elif winner == Player.black:
    print("Black win")
  else:
    print("White win")

if __name__ == '__main__':
  main()
