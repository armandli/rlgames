import time
from gotypes import Player
from goboard import GameState
from agents.random import RandomBot
from util import print_board, print_move

def main():
  board_size = 9
  game = GameState.new_game(board_size)
  bots = {
    Player.black : RandomBot(),
    Player.white : RandomBot(),
  }
  while not game.is_over():
    time.sleep(0.3) # slow down so we can observe

    #print(chr(27) + "[2J") #clear screen
    print_board(game.board)
    bot_move = bots[game.nplayer].select_move(game)
    print_move(game.nplayer, bot_move)
    game = game.apply_move(bot_move)

if __name__ == '__main__':
  main()
