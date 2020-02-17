import numpy as np

from keras.optimizers import SGD

from rlgames.encoders.base import get_encoder_by_name
from rlgames.kerasutil import load_model_from_hdf5_group, save_model_to_hdf5_group
from rlgames.encoders.base import Encoder
from rlgames.agents.base import Agent
from rlgames.rl.zero import ZeroExperienceCollector

# alpha go zero

# TODO: model should add batch normalization layer
# TODO: model should use residual net layer model

class Branch:
  def __init__(self, prior):
    self.prior = prior
    self.visit_count = 0
    self.total_value = 0.

class ZeroTreeNode:
  def __init__(self, state, value, priors, parent, last_move):
    self.state = state
    self.value = value
    self.parent = parent
    self.last_move = last_move
    self.total_visit_count = 1
    self.branches = {}
    for move, prior in priors.items():
      if state.is_valid_move(move):
        self.branches[move] = Branch(prior)
    self.children = {}

  def moves(self):
    return self.branches.keys()

  def add_child(self, move, child_node):
    self.children[move] = child_node

  def has_child(self, move):
    if move is None:
      return False
    return move in self.children

  def get_child(self, move):
    return self.children[move]

  def expected_value(self, move):
    branch = self.branches[move]
    if branch.visit_count == 0:
      return 0.
    return branch.total_value / branch.visit_count

  def prior(self, move):
    return self.branches[move].prior

  def visit_count(self, move):
    if move in self.branches:
      return self.branches[move].visit_count
    return 0

  def record_visit(self, move, value):
    self.total_visit_count += 1
    self.branches[move].visit_count += 1
    self.branches[move].total_value += value

# no special logic on when to pass, this is because pass is part of the natural choices
class ZeroAgent(Agent):
  def __init__(self, model, encoder, num_rounds=1600, exploration_factor=2.0):
    super().__init__()
    self.model = model
    self.encoder = encoder
    self.collector = None
    self.c = exploration_factor
    self.num_rounds = num_rounds

  def select_move(self, gs):
    root = self.create_node(gs)
    for i in range(self.num_rounds): #self play used 1600 rounds
      #find a known leaf
      node = root
      next_move = self.select_branch(node)
      while node.has_child(next_move):
        node = node.get_child(next_move)
        next_move = self.select_branch(node)
      # if we cannot expand any children, update the visit count,
      # we cannot choose a different node to visit because visit
      # count indicate best choice.
      if next_move is None:
        value = -1. * node.value
        move = node.last_move
        node = node.parent
        while node is not None:
          node.record_visit(move, value)
          move = node.last_move
          node = node.parent
          value = -1. * value
      else:
        #expand the new leaf and update the tree
        new_state = node.state.apply_move(next_move)
        child_node = self.create_node(new_state, next_move, node)
        #update tree all the way to the root
        move = next_move
        value = -1. * child_node.value
        while node is not None:
          node.record_visit(move, value)
          move = node.last_move
          node = node.parent
          value = -1. * value
    #collect experience, for AlphaGo Zero, collect the visit count
    if self.collector is not None:
      root_state_tensor = self.encoder.encode(gs)
      visit_counts = np.array([root.visit_count(self.encoder.decode_move_index(idx)) for idx in range(self.encoder.num_moves())])
      self.collector.record_decision(root_state_tensor, visit_counts)
    #select a move, by picking the immediate child with the highest visit count
    return max(root.moves(), key=root.visit_count)

  def select_branch(self, node):
    all_moves = node.moves()
    if len(all_moves) == 0:
      return None
    total_n = node.total_visit_count
    noises = np.random.dirichlet([0.03 for _ in range(len(all_moves))])
    moves_with_noises = [(move, noise) for move, noise in zip(all_moves, noises)]
    def score_branch(move_and_noise):
      move = move_and_noise[0]
      noise = move_and_noise[1]
      q = node.expected_value(move)
      p = node.prior(move)
      n = node.visit_count(move)
      #TODO: is this the right way to apply dirichlet noise ?
      return q + (self.c + noise) * p * (np.sqrt(total_n) / (n + 1))
    return max(moves_with_noises, key=score_branch)[0]

  def create_node(self, gs, move=None, parent=None):
    state_tensor = self.encoder.encode(gs)
    model_input = np.array([state_tensor])
    priors, values = self.model.predict(model_input)
    priors = priors[0]
    value = values[0][0]
    move_priors = {
      self.encoder.decode_move_index(idx): p
      for idx, p in enumerate(priors)
    }
    new_node = ZeroTreeNode(gs, value, move_priors, parent, move)
    if parent is not None:
      parent.add_child(move, new_node)
    return new_node

  def set_collector(self, collector):
    self.collector = collector

  def train(self, experience, learning_rate, batch_size):
    num_examples = experience.states.shape[0]
    model_input = experience.states
    visit_sums = np.sum(experience.vcounts, axis=1).reshape((num_examples, 1))
    action_target = experience.vcounts / visit_sums
    value_target = experience.rewards
    self.model.compile(SGD(lr=learning_rate), loss=['categorical_crossentropy', 'mse'])
    self.model.fit(model_input, [action_target, value_target], batch_size=batch_size)

  def serialize(self, h5file):
    h5file.create_group('encoder')
    h5file['encoder'].attrs['name'] = self.encoder.name()
    h5file['encoder'].attrs['board_sz'] = self.encoder.sz
    h5file.create_group('model')
    save_model_to_hdf5_group(self.model, h5file['model'])

def load_zero_agent(h5file, eval_rounds = 1600, exploration_factor=2.):
  model = load_model_from_hdf5_group(h5file['model'])
  encoder_name = h5file['encoder'].attrs['name']
  if not isinstance(encoder_name, str):
    encoder_name = encoder_name.decode('ascii')
  board_sz = h5file['encoder'].attrs['board_sz']
  encoder = get_encoder_by_name(encoder_name, board_sz)
  return ZeroAgent(model, encoder, eval_rounds, exploration_factor)
