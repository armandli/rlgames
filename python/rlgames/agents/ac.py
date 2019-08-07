import numpy as np
import h5py

from keras.optimizers import SGD

from rlgames.game_base import Move
from rlgames.goboard import GameState
from rlgames.kerasutil import load_model_from_hdf5_group, save_model_to_hdf5_group
from rlgames.agents.base import Agent
from rlgames.agents.helper import is_point_an_eye
from rlgames.encoders.base import get_encoder_by_name

class ACAgent(Agent):
  def __init__(self, model, encoder):
    Agent.__init__(self)
    self.model = model
    self.encoder = encoder
    self.collector = None
    self.temperature = 1.0

  def select_move(self, gs):
    num_moves = self.encoder.num_points()
    board_tensor = self.encoder.encode(gs)
    X = np.array([board_tensor])
    actions, values = self.model.predict(X)
    move_probs = actions[0]
    estimated_value = values[0][0]
    eps = 1e-6
    #TODO: how to use temperature here?
    #TODO: move this into a function that does the clipping
    move_probs = np.clip(move_probs, eps, 1 - eps)
    move_probs = move_probs / np.sum(move_probs)

    candidates = np.arange(num_moves)
    ranked_moves = np.random.choice(candidates, num_moves, replace=False,
      p=move_probs)
    for point_idx in ranked_moves:
      point = self.encoder.decode_point_index(point_idx)
      move = Move.play(point)
      is_move_valid=_valid = gs.is_valid_move(move)
      is_eye = is_point_an_eye(gs.board, point, gs.nplayer)
      if is_move_valid and not is_eye:
        if self.collector is not None:
          self.collector.record_decision(state=board_tensor,
                  action=point_idx, estimated_value=estimated_value)
        return Move.play(point)
    return Move.pass_turn()

  def train(self, experience, lr=0.1, batch_size=128):
    opt = SGD(lr=lr)
    # loss is bigger from value function, reducing its loss weight by half
    # we can choose different loss function for different output of the
    # same network
    self.model.compile(optimizer=opt, loss=['categorical_crossentropy', 'mse'], loss_weights=[1.0, 0.5])
    n = experience.states.shape[0]
    num_moves = self.encoder.num_points()
    policy_target = np.zeros((n, num_moves))
    value_target = np.zeros((n,))
    for i in range(n):
      action = experience.actions[i]
      policy_target[i][action] = experience.advantages[i]
      reward = experience.rewards[i]
      value_target[i] = reward
    self.model.fit(experience.states, [policy_target, value_target],
      batch_size=batch_size, epochs=1)

  def serialize(self, h5file):
    h5file.create_group('encoder')
    h5file['encoder'].attrs['name'] = self.encoder.name()
    h5file['encoder'].attrs['board_sz'] = self.encoder.sz
    h5file.create_group('model')
    save_model_to_hdf5_group(self.model, h5file['model'])

  def set_temperature(self, temp):
    self.temperature = temp

  def set_collector(self, collector):
    self.collector = collector

def load_ac_agent(h5file):
  model = load_model_from_hdf5_group(h5file['model'])
  encoder_name = h5file['encoder'].attrs['name']
  if not isinstance(encoder_name, str):
    encoder_name = encoder_name.decode('ascii')
  board_sz = h5file['encoder'].attrs['board_sz']
  encoder = get_encoder_by_name(encoder_name, board_sz)
  return ACAgent(model, encoder)
