from rlgames.common_types import Player
from rlgames.game_base import Move
from rlgames.tictactoe_board import GameState
from rlgames.agents.minimax import MinimaxAgent
from rlgames.agents.mcts import MCTSAgent
from rlgames.util import print_board, print_move, point_from_coord

def main():
  board_size = 3
  game = GameState.new_game(board_size)
  #bot = MinimaxAgent(5, None)
  bot = MCTSAgent(362000, 0.1, 64)
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
  print_board(game.board)
  winner = game.winner()
  if winner is None:
    print("Tie")
  elif winner == Player.black:
    print("Human won")
  else:
    print("Bot won")

if __name__ == '__main__':
  main()
