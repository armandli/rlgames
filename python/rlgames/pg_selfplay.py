import os
import argparse
import h5py

from rlgames.common_types import Point, Player
from rlgames.game_base import Move
from rlgames.goboard import GameState
from rlgames.encoders.base import get_encoder_by_name
from rlgames.agents.pg import PolicyAgent, load_policy_agent
from rlgames.rl.experience import ExperienceCollector, combine_experience

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--board-size', '-s', type=int, default=5)
  parser.add_argument('--model-size', '-m', type=str, default='small')
  parser.add_argument('--encoder-name', '-e', type=str, default='sixplane')
  parser.add_argument('--data-dir', '-d', type=str, default='/home/armandli/rlgames/data')
  parser.add_argument('--learning-rate', '-l', type=float, default=0.00001)
  parser.add_argument('--batchsize', '-b', type=int, default='2048')
  parser.add_argument('--rounds', '-r', type=int, default=1000000)
  return parser.parse_args()

def load_agent(agent_path, args):
  if not os.path.isfile(agent_path):
    #if agent does not exist yet, create a new one, and save it
    encoder = get_encoder_by_name(args.encoder_name, args.board_size)
    if args.model_size == 'small':
      from rlgames.rl_models.pg_small import pg_model
      model = pg_model(encoder)
    else:
      raise ValueError('Unknown model size: {}'.format(args.model_size))
    agent = PolicyAgent(model, encoder)
    agent.serialize(h5py.File(agent_path, 'w'))
  agent = load_policy_agent(h5py.File(agent_path, 'r'))
  return agent

def setup_agent(agent_path, args):
  agent = load_agent(agent_path, args)
  collector = ExperienceCollector()
  agent.set_collector(collector)
  return agent, collector

def temperature_decay(temp, round_no):
  return temp * 0.9999

def eval_new_agent(args, new_agent, old_agent, rounds = 100, acceptability = 0.53):
  old_temp = new_agent.temperature
  new_temp = old_agent.temperature

  new_agent.set_temperature(0.)
  old_agent.set_temperature(0.)

  players = {
    Player.black : new_agent,
    Player.white : old_agent,
  }
  win_count = 0
  for _ in range(int(rounds / 2)):
    game = GameState.new_game(args.board_size)
    while not game.is_over():
      move = players[game.nplayer].select_move(game)
      game = game.apply_move(move)
    winner = game.winner()
    if winner == Player.black:
      win_count += 1
  players = {
    Player.black : old_agent,
    Player.white : new_agent,
  }
  for _ in range(int(rounds / 2)):
    game = GameState.new_game(args.board_size)
    while not game.is_over():
      move = players[game.nplayer].select_move(game)
      game = game.apply_move(move)
    winner = game.winner()
    if winner == Player.white:
      win_count += 1

  new_agent.set_temperature(new_temp)
  old_agent.set_temperature(old_temp)

  print('win count: {}'.format(win_count))

  if float(win_count) > rounds * acceptability:
    return True
  else:
    return False

def main():
  args = parse_args()
  temperature = 0.3
  agent_path = args.data_dir + '/agents/pg_' + args.model_size + '_' + args.encoder_name + '_' + str(args.board_size) + '.h5'
  print('Agent path: {}'.format(agent_path))
  agent1, collector1 = setup_agent(agent_path, args)
  agent2, collector2 = setup_agent(agent_path, args)
  players = {
    Player.black : agent1,
    Player.white : agent2,
  }
  for i in range(args.rounds):
    round_no = i + 1
    temperature = temperature_decay(temperature, round_no)
    if i % 10 == 0:
      print('Begin round {} selfplay. temperature {}'.format(round_no, temperature))
    agent1.set_temperature(temperature)
    agent2.set_temperature(temperature)
    collector1.begin_episode()
    collector2.begin_episode()
    game = GameState.new_game(args.board_size)
    while not game.is_over():
      move = players[game.nplayer].select_move(game)
      game = game.apply_move(move)
    winner = game.winner()
    if winner == Player.black:
      collector1.complete_episode(1.)
      collector2.complete_episode(-1.)
    else:
      collector1.complete_episode(-1.)
      collector2.complete_episode(1.)
    # learng and evaluate the experience
    if round_no % 1000 == 0:
      print('Begin round {} training'.format(round_no))
      exp = combine_experience([collector1, collector2])
      agent1.train(exp, args.learning_rate, batchsize=args.batchsize)
      print('Training complete.')
      # not worrying about improvement anymore because policy gradient has high
      # variance, difficult to improve in the begining
      agent1.serialize(h5py.File(agent_path, 'w'))
      agent2 = load_policy_agent(h5py.File(agent_path, 'r'))
      agent2.set_collector(collector2)
      collector1.clear()
      collector2.clear()
  agent1.serialize(h5py.File(agent_path, 'w'))
  print('PG Selfplay complete. agent is in {}'.format(agent_path))

if __name__ == '__main__':
  main()
