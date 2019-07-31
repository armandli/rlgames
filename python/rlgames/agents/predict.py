import numpy as np
import h5py

from rlgames.agents.base import Agent
from rlgames.agents.helper import is_point_an_eye
from rlgames.game_base import Move
from rlgames.goboard import Board
from rlgames.encoders import get_encoder_by_name
from rlgames.kerasutil import save_model_to_hdf5_group, load_model_from_hdf5_group

class DeepLearningAgent(Agent):
  def __init__(self, model, encoder):
    Agent.__init__(self)
    self.model = model
    self.encoder = encoder

  def select_move(self, gs):
    eps = 1e-6
    num_actions = self.encoder.num_points
    move_probs = self.predict(gs)
    #scaling, clipping and re-normalizing move probabilities to reduce ambiguity,
    #then sample from the rescaled moves
    move_probs = move_probs ** 3
    move_probs = np.clip(move_probs, eps, 1 - eps)
    move_probs = move_probs / np.sum(move_probs)
    candidates = np.arange(num_actions)
    actions = np.random.choice(candidates, num_actions, replace=False, p=move_probs)
    for point_idx in actions:
      point = self.encoder.decode_point_inde(point_idx)
      if gs.is_valid_move(Move.play(point)) and not is_point_an_eye(gs.board, point, gs.nplayer):
        return Move.play(point)
    return Move.pass_turn()

  def predict(self, gs):
    estate = self.encoder.encode(gs)
    x = np.array([estate])
    x = x.astype('float32')
    return self.model.predict(x)[0]

  def serialize(self, h5file):
    h5file.create_group('encoder')
    h5file['encoder'].attrs['name'] = self.encoder.name()
    h5file['encoder'].attrs['board_sz'] = self.encoder.sz
    h5file.create_group('model')
    save_model_to_hdf5_group(self.model, h5file['model'])

def load_prediction_agent(h5file):
  model = load_model_from_hdf5_group(h5file['model'])
  encoder_name = h5file['encoder'].attrs['name']
  if not isinstance(encoder_name, str):
    encoder_name = encoder_name.decode('ascii')
  board_sz = h5file['encoder'].attrs['board_sz']
  encoder = get_encoder_by_name(encoder_name, board_sz)
  return DeepLearningAgent(model, encoder)
