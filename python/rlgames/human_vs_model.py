import os
import argparse
import h5py

from rlgames.common_types import Player
from rlgames.game_base import Move
from rlgames.goboard import GameState
from rlgames.agents.pg import load_policy_agent
from rlgames.agents.q import load_q_agent
from rlgames.agents.ac import load_ac_agent
from rlgames.agents.predict import load_prediction_agent
from rlgames.util import print_board, print_move, point_from_coord

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--agent-type', '-t', type=str, required=True)
  parser.add_argument('--board-size', '-s', type=int, required=True)
  parser.add_argument('--encoder', '-e', type=str, required=True)
  parser.add_argument('--model', '-m', type=str, required=True)
  parser.add_argument('--playas', '-p', type=str, default='black')
  args = parser.parse_args()
  return args

def load_agent(args):
  if args.agent_type == 'pg':
    agent_path = '/home/armandli/rlgames/data/agents/pg_' + args.model + '_' + args.encoder + '_' + str(args.board_size) + '.h5'
    if not os.path.isfile(agent_path):
      raise ValueError('PG Agent {} does not exist!'.format(agent_path))
    agent = load_policy_agent(h5py.File(agent_path, 'r'))
  elif args.agent_type == 'q':
    agent_path = '/home/armandli/rlgames/data/agents/q_' + args.model + '_' + args.encoder + '_' + str(args.board_size) + '.h5'
    if not os.path.isfile(agent_path):
      raise ValueError('Q Learning Agent {} does not exist!'.format(agent_path))
    agent = load_q_agent(h5py.File(agent_path, 'r'))
  elif args.agent_type == 'ac':
    agent_path = '/home/armandli/rlgames/data/agents/ac_' + args.model + '_' + args.encoder + '_' + str(args.board_size) + '.h5'
    if not os.path.isfile(agent_path):
      raise ValueError('AC Learning Agent {} does not exist!'.format(agent_path))
    agent = load_ac_agent(h5py.File(agent_path, 'r'))
  elif args.agent_type == 'sl':
    agent_path = '/home/armandli/rlgames/data/agents/sl_' + args.model + '_' + args.encoder + '_' + str(args.board_size) + '.h5'
    if not os.path.isfile(agent_path):
      raise ValueError('SL Agent {} does not exist!'.format(agent_path))
    agent = load_prediction_agent(h5py.File(agent_path, 'r'))
  else:
    raise ValueError('Unknown agent type {}',format(args.agent_type))
  return agent

def main():
  args = parse_args()
  agent = load_agent(args)
  board_size = args.board_size
  game = GameState.new_game(board_size)
  if args.playas == 'black':
    human_play = Player.black
  elif args.playas == 'white':
    human_play = Player.white
  else:
    raise ValueError('Unknown option for playas: {}'.format(args.playas))
  while not game.is_over():
    print_board(game.board)
    if game.nplayer == human_play:
      human_move = input('-- ')
      if len(human_move) > 1:
        point = point_from_coord(human_move.strip())
        move = Move.play(point)
      else:
        move = Move.pass_turn()
    else:
      move = agent.select_move(game)
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
