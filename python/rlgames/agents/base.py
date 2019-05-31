from abc import abstractmethod

class Agent:
  @abstractmethod
  def select_move(self, game_state):
    raise NotImplementedError()
