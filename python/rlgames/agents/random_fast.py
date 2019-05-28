import numpy as np
from rlgames.agents.base import Agent
from rlgames.agents.helper import is_point_an_eye
from rlgames.game_base import Move
from rlgames.common_types import Point

class FastRandomAgent(Agent):
  def __init__(self, sz):
    super().__init__()
    self.cache = self.init_cache_(sz)

  def init_cache_(self, sz):
    return [Point(ri, ci) for ri in range(1, sz + 1) for ci in range(1, sz + 1)]

  def select_move(self, game_state):
    idxes = np.arange(len(self.cache))
    np.random.shuffle(idxes)
    for idx in idxes:
      m = Move.play(self.cache[idx])
      if (game_state.is_valid_move(m) and 
          not is_point_an_eye(game_state.board, m.pt, game_state.nplayer)):
        return m
    return Move.pass_turn()
