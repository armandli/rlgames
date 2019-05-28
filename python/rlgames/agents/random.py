import random
from rlgames.agents.base import Agent
from rlgames.agents.helper import is_point_an_eye
from rlgames.game_base import Move
from rlgames.common_types import Point

class RandomAgent(Agent):
  def select_move(self, game_state):
    """choose a random valid move that preserves its own eyes."""
    candidates = list()
    for move in game_state.legal_moves():
      if not move.is_pass and not move.is_resign:
        if not is_point_an_eye(game_state.board, move.pt, game_state.nplayer):
          candidates.append(move)
    if not candidates:
      return Move.pass_turn()
    return random.choice(candidates)
