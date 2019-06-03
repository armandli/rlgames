import math
import random
import numpy as np

from rlgames.agents.base import Agent
from rlgames.agents.helper import is_point_an_eye
from rlgames.game_base import Move
from rlgames.common_types import Player, Point

#TODO: add resign policy, when pct winning is lower than 10%

class MCTSNode:
  def __init__(self, game_state, parent=None, move=None):
    self.game_state = game_state
    self.parent = parent
    self.pmove = move #previous move
    self.qvalue = 0.
    self.ncount = 0.
    self.children  = set()                    #explored children
    self.pchildren = set()                    #explored children with unexplored leaf
    self.uchildren = game_state.legal_moves() #unexplored children

  def add_child(self, node):
    self.uchildren.remove(node.pmove)
    self.children.add(node)
    self.pchildren.add(node)

  def remove_pchild(self, node):
    self.pchildren.remove(node)
    if self.parent is not None and not self.pchildren and not self.uchildren:
      self.parent.remove_pchild(self)

  def update(self, qvalue, count):
    self.qvalue += qvalue
    self.ncount += count
    if self.parent is not None:
      self.parent.update(qvalue, count)

#TODO: put these somewhere common
MAX_SCORE = 10.
MIN_SCORE = -10.
TIE_SCORE = 0.

class MCTSAgent(Agent):
  def __init__(self, expandmax, exploration_factor = 1., mc_sample_size = 1):
    super().__init__()
    self.expandmax = expandmax
    self.efactor = exploration_factor
    self.mc_trials = mc_sample_size
    self.cache = []

  def select_move(self, game_state):
    if len(self.cache) != game_state.board.sz * game_state.board.sz:
      self.cache = self.init_cache_(game_state.board.sz)
    root = MCTSNode(game_state)
    for _ in range(self.expandmax):
      new_node = self.recursive_uct_(root)
      if new_node is None:
        break
      qvalue = self.mc_play_(new_node.game_state)
      new_node.update(qvalue, self.mc_trials)
    best_nodes = list()
    best_score = self.init_best_score_(root.game_state.nplayer)
    for child in root.children:
      score = child.qvalue / child.ncount
      if self.is_improvement_(score, best_score, root.game_state.nplayer):
        best_score = score
        best_nodes = [child]
      elif score == best_score:
        best_nodes.append(child)
    if best_nodes:
      return random.choice(best_nodes).pmove
    else:
      return Move.pass_turn()

  def init_best_score_(self, player):
    if player == Player.black:
      return MIN_SCORE
    else:
      return MAX_SCORE

  def is_improvement_(self, score, best_score, player):
    if player == Player.black:
      return score > best_score
    else:
      return score < best_score

  def recursive_uct_(self, node):
    if node.uchildren:
      new_move = random.choice(node.uchildren)
      new_game_state = node.game_state.apply_move(new_move)
      new_node = MCTSNode(new_game_state, node, new_move)
      node.add_child(new_node)
      if new_game_state.is_over():
        node.remove_pchild(new_node)
      return new_node
    else:
      best_children = list()
      best_score = self.init_best_score_(node.game_state.nplayer)
      for child in node.pchildren:
        score = child.qvalue / child.ncount
        explore_factor = self.efactor * math.sqrt(2 * math.log(node.ncount) / child.ncount)
        if child.game_state.nplayer == Player.black:
          score += explore_factor
        else:
          score -= explore_factor
        if self.is_improvement_(score, best_score, node.game_state.nplayer):
          best_score = score
          best_children = [child]
        elif score == best_score:
          best_children.append(child)
      if best_children:
        return self.recursive_uct_(random.choice(best_children))
      else:
        return None

  def init_cache_(self, sz):
    return [Point(ri, ci) for ri in range(1, sz + 1) for ci in range(1, sz + 1)]

  def random_move_(self, game_state):
    idxes = np.arange(len(self.cache))
    np.random.shuffle(idxes)
    for idx in idxes:
      m = Move.play(self.cache[idx])
      if (game_state.is_valid_move(m) and
          not is_point_an_eye(game_state.board, m.pt, game_state.nplayer)):
        return m
    return Move.pass_turn()

  def mc_play_(self, game_state):
    score = 0.
    for _ in range(self.mc_trials):
      play_state = game_state
      while not play_state.is_over():
        move = self.random_move_(play_state)
        play_state = play_state.apply_move(move)
      winner = play_state.winner()
      if winner is None:
        score += TIE_SCORE
      elif winner == Player.black:
        score += MAX_SCORE
      else:
        score += MIN_SCORE
    return score
