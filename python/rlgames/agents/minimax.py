import random
from agents.base import Agent
from game_base import Move
from common_types import Point, Player

#TODO: put these somewhere common
MAX_SCORE = 1.
MIN_SCORE = -1.
TIE_SCORE = 0.

class MinimaxAgent(Agent):
  def __init__(self, depth, eval_fn):
    super().__init__()
    self.depth = depth
    self.eval_fn = eval_fn

  #TODO: there could be multiple equally good moves, we should allow random choice
  def select_move(self, game_state):
    #(_, best_move) = self.recursive_minimax_search_(game_state, self.depth, self.eval_fn)
    (_, best_move) = self.recursive_alpha_beta_minimax_search_(game_state, self.depth, self.eval_fn, MIN_SCORE, MAX_SCORE)
    return best_move

  def is_improvement_(self, score, best_score, player):
    if player == Player.black:
      return score > best_score
    else:
      return score < best_score

  def recursive_minimax_search_(self, game_state, depth, eval_fn):
    if game_state.is_over():
      winner = game_state.winner()
      if winner == Player.black:
        return (MAX_SCORE, None)
      elif winner is None:
        return (TIE_SCORE, None)
      else:
        return (MIN_SCORE, None)
    if depth == 0:
      return (eval_fn(game_state), None)
    best_move = Move.pass_turn()
    if game_state.nplayer == Player.black:
      best_score = MIN_SCORE
    else:
      best_score = MAX_SCORE
    for m in game_state.legal_moves():
      nstate = game_state.apply_move(m)
      ndepth = depth
      if nstate.nplayer == Player.black:
        ndepth -= 1
      (score, _) = self.recursive_minimax_search_(nstate, ndepth, eval_fn)
      if self.is_improvement_(score, best_score, game_state.nplayer):
        best_score = score
        best_move = m
    return (best_score, best_move)

  def should_prune_(self, score, alpha, beta, player):
    if player == Player.black:
      return score > beta
    else:
      return score < alpha

  def recursive_alpha_beta_minimax_search_(self, game_state, depth, eval_fn, alpha, beta):
    """
    Alpha-Beta Pruning: introduce new parameters alpha and beta.
    alpha is the best value available to the maximizer from the parent to the root
    beta is the best value available to the minimizer from the parent to the root
    we use those 2 values to prune true segments downstream.
    the idea is if the current node is a maximizer, and the beta it has received from its parent (minimizer)
    has a lower value than the maximum value found by the current node, we no longer need to explore downstream.
    if the current node is a minimizer, and the alpha it has received from its parent (maximizer)
    has a higher value than the minimum value found by the current node, we no longer need to explore downstream.
    in this implementation, we don't prune on equality
    """
    if game_state.is_over():
      winner = game_state.winner()
      if winner == Player.black:
        return (MAX_SCORE, None)
      elif winner is None:
        return (TIE_SCORE, None)
      else:
        return (MIN_SCORE, None)
    if depth == 0:
      return (eval_fn(game_state), None)
    best_move = Move.pass_turn()
    if game_state.nplayer == Player.black:
      best_score = MIN_SCORE
    else:
      best_score = MAX_SCORE
    for m in game_state.legal_moves():
      if self.should_prune_(best_score, alpha, beta, game_state.nplayer):
        break
      nstate = game_state.apply_move(m)
      ndepth = depth
      if nstate.nplayer == Player.black:
        ndepth -= 1
      (score, _) = self.recursive_alpha_beta_minimax_search_(nstate, ndepth, eval_fn, alpha, beta)
      if self.is_improvement_(score, best_score, game_state.nplayer):
        best_score = score
        best_move = m
        if game_state.nplayer == Player.black:
          if best_score > alpha:
            alpha = best_score
        else:
          if best_score < beta:
            beta = best_score
    return (best_score, best_move)
