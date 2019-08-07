import numpy as np
import h5py

from keras.optimizers import SGD

from rlgames.game_base import Move
from rlgames.goboard import GameState
from rlgames.kerasutil import load_model_from_hdf5_group, save_model_to_hdf5_group
from rlgames.agents.base import Agent
from rlgames.agents.helper import is_point_an_eye
from rlgames.encoders.base import get_encoder_by_name

#TODO: try active Q learning
#TODO: try double-Q learning

class QAgent(Agent):
  def __init__(self, model, encoder):
    self.model = model
    self.encoder = encoder
    self.collector = None
    self.temperature = 0.

  def select_move(self, gs):
    board_tensor = self.encoder.encode(gs)
    moves = []
    board_tensors = []
    for move in gs.legal_moves():
      if not move.is_play:
        continue
      moves.append(self.encoder.encode_point(move.pt))
      board_tensors.append(board_tensor)
    if not moves:
      return Move.pass_turn()
    num_moves = len(moves)
    board_tensors = np.array(board_tensors)
    move_vectors = np.zeros((num_moves, self.encoder.num_points()))
    for i, move in enumerate(moves):
      move_vectors[i][move] = 1.
    #Q-learning uses 2 input tensors: the states and the actions
    values = self.model.predict([board_tensors, move_vectors])
    values = values.reshape(len(moves))
    ranked_moves = self.rank_move_eps_greedy(values)
    for move_idx in ranked_moves:
      point = self.encoder.decode_point_index(moves[move_idx])
      if not is_point_an_eye(gs.board, point, gs.nplayer):
        if self.collector is not None:
          self.collector.record_decision(state=board_tensor, action=moves[move_idx])
        return Move.play(point)
    return Move.pass_turn()

  def rank_move_eps_greedy(self, values):
    if np.random.random() < self.temperature:
      #replace real value with random value for exploration
      values = np.random.random(values.shape)
    rank_moves = np.argsort(values)
    return rank_moves[::-1] #reverse the list for values from highest to lowest

  def train(self, experience, lr=0.1, batch_size=128):
    opt = SGD(lr=lr)
    #uses MSE because we want to predict the exact goodness value of the state-value
    self.model.compile(loss='mse', optimizer=opt)
    n = experience.states.shape[0]
    num_moves = self.encoder.num_points()
    y = np.zeros((n,))
    actions = np.zeros((n, num_moves))
    for i in range(n):
      action = experience.actions[i]
      reward = experience.rewards[i]
      actions[i][action] = 1
      y[i] = reward
    self.model.fit([experience.states, actions], y, batch_size=batch_size, epochs=1)

  def set_temperature(self, temp):
    self.temperature = temp

  def set_collector(self, collector):
    self.collector = collector

  def serialize(self, h5file):
    h5file.create_group('encoder')
    h5file['encoder'].attrs['name'] = self.encoder.name()
    h5file['encoder'].attrs['board_sz'] = self.encoder.sz
    h5file.create_group('model')
    save_model_to_hdf5_group(self.model, h5file['model'])

def load_q_agent(h5file):
  model = load_model_from_hdf5_group(h5file['model'])
  encoder_name = h5file['encoder'].attrs['name']
  board_sz = h5file['encoder'].attrs['board_sz']
  encoder = get_encoder_by_name(encoder_name, board_sz)
  return QAgent(model, encoder)
