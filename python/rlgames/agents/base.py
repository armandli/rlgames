from abc import abstractmethod

class Agent(object):
  @abstractmethod
  def select_move(self, game_state):
    raise NotImplementedError()
