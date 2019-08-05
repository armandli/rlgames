import argparse
import h5py

from rlgames.common_types import Player
from rlgames.game_base import Move
from rlgames.goboard import GameState
from rlgames.agents.predict import DeepLearningAgent, load_prediction_agent
from rlgames.util import print_board, print_move, point_from_coord

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--agent', '-a', type=str,
          default='/home/armandli/rlgames/data/agents/imitation_model_medium_sevenplane.h5')
  args = parser.parse_args()
  return args

def main():
  args = parse_args()
  board_size = 19
  game = GameState.new_game(board_size)
  bot = load_prediction_agent(h5py.File(args.agent, 'r'))
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
