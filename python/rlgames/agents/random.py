import random
from agents.base import Agent
from agents.helper import is_point_an_eye
from goboard import Move
from gotypes import Point

class RandomBot(Agent):
  def select_move(self, game_state):
    """choose a random valid move that preserves its own eyes."""
    candidates = []
    for r in range(1, game_state.board.nrows + 1):
      for c in range(1, game_state.board.ncols + 1):
        candidate = Point(r, c)
        if game_state.is_valid_move(Move.play(candidate)) and \
           not is_point_an_eye(game_state.board, candidate, game_state.nplayer):
          candidates.append(candidate)
    if not candidates:
      return Move.pass_turn()
    return Move.play(random.choice(candidates))
