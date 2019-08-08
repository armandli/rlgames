import os
import argparse
import h5py

from rlgames.common_types import Player
from rlgames.game_base import Move
from rlgames.goboard import GameState
from rlgames.agents.random_fast import FastRandomAgent
from rlgames.agents.pg import load_policy_agent
from rlgames.agents.q import load_q_agent
from rlgames.agents.ac import load_ac_agent
from rlgames.agents.zero import load_zero_agent
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
  elif args.agent_type == 'zero':
    agent_path = '/home/armandli/rlgames/data/agents/zero_' + args.model + '_' + args.encoder + '_' + str(args.board_size) + '.h5'
    if not os.path.isfile(agent_path):
      raise ValueError('AlphaGo Zero Learning Agent {} does not exist!'.format(agent_path))
    agent = load_zero_agent(h5py.File(agent_path, 'r'))
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
  board_size = args.board_size
  agent = load_agent(args)
  random = FastRandomAgent(args.board_size)
  if args.playas == 'black':
    random_play = Player.black
  elif args.playas == 'white':
    random_play = Player.white
  else:
    raise ValueError('Unknown option for playas: {}'.format(args.playas))
  win_count = 0
  for  _ in range(1000):
    game = GameState.new_game(board_size)
    while not game.is_over():
      if game.nplayer == random_play:
        move = random.select_move(game)
      else:
        move = agent.select_move(game)
      game = game.apply_move(move)
    winner = game.winner()
    if winner is not None and winner != random_play:
      win_count += 1
  print('Model won: {}/1000'.format(win_count))

if __name__ == '__main__':
  main()
