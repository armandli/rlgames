import h5py
import numpy as np

from keras.models import Sequential
from keras.optimizers import SGD

from rlgames.agents.base import Agent
from rlgames.agents.helper import is_point_an_eye
from rlgames.game_base import Move
from rlgames.encoders.base import get_encoder_by_name
from rlgames.kerasutil import save_model_to_hdf5_group, load_model_from_hdf5_group
from rlgames.rl.experience import ExperienceCollector

class PolicyAgent(Agent):
  def __init__(self, model, encoder):
    self.model = model
    self.encoder = encoder
    self.collector = None
    self.temperature = 0.

  def select_move(self, gs):
    num_moves = self.encoder.num_points()
    board_tensor = self.encoder.encode(gs)
    X = np.array([board_tensor])
    # epsilon greedy exploration
    if np.random.random() < self.temperature:
      move_probs = np.ones(num_moves) / num_moves
    else:
      move_probs = self.model.predict(X)[0]
      move_probs = self.clip_probs_(move_probs)
    candidates = np.arange(num_moves)
    ranked_moves = np.random.choice(candidates, num_moves, replace=False, p=move_probs)
    for pt_idx in ranked_moves:
      point = self.encoder.decode_point_index(pt_idx)
      move = Move.play(point)
      is_valid = gs.is_valid_move(move)
      is_eye = is_point_an_eye(gs.board, point, gs.nplayer)
      if is_valid and not is_eye:
        if self.collector is not None:
          self.collector.record_decision(state=board_tensor, action=pt_idx)
        return move
    #this policy gradient does not learn from passing turn
    return Move.pass_turn()

  def serialize(self, h5file):
    h5file.create_group('encoder')
    h5file['encoder'].attrs['name'] = self.encoder.name()
    h5file['encoder'].attrs['board_sz'] = self.encoder.sz
    h5file.create_group('model')
    save_model_to_hdf5_group(self.model, h5file['model'])

  def set_collector(self, collector):
    self.collector = collector

  def set_temperature(self, temp):
    self.temperature = temp

  def clip_probs_(self, probs):
    min_p = 1e-5
    max_p = 1 - min_p
    c = np.clip(probs, min_p, max_p)
    return c / np.sum(c)

  def prepare_experience_data_(self, experience, board_sz):
    experience_size = experience.states.shape[0]
    target_vectors = np.zeros((experience_size, board_sz * board_sz))
    for i in range(experience_size):
      action = experience.actions[i]
      reward = experience.rewards[i]
      target_vectors[i][action] = reward
    return target_vectors

  def train(self, experience, lr, clipnorm=1.0, batchsize=512):
    self.model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=lr, clipnorm=clipnorm))
    target_vectors = self.prepare_experience_data_(experience, self.encoder.sz)
    # batch size is important for PG to learn faster
    self.model.fit(experience.states, target_vectors, batch_size=batchsize, epochs=1)

def load_policy_agent(h5file):
  model = load_model_from_hdf5_group(h5file['model'])
  encoder_name = h5file['encoder'].attrs['name']
  board_sz = h5file['encoder'].attrs['board_sz']
  encoder = get_encoder_by_name(encoder_name, board_sz)
  return PolicyAgent(model, encoder)
