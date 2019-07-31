from rlgames.game_base import Move
from rlgames.goboard import GameState

class TerminationStrategy:
  def __init__(self):
    pass
  def should_pass(self, gs):
    return False
  def should_resign(self, gs):
    return False

class PassWhenOpponentPasses(TerminationStrategy):
  def should_pass(self, gs):
    if gs.pmove is not None:
      return True if gs.pmove.is_pass else False

class ResignLargeMargin(TerminationStrategy):
  def should_pass(self, gs):
    return False
  def should_resign(self, gs):
    #TODO
    return False

def get(termination):
  if termination == 'opponent_passes':
    return PassWhenOpponentPasses()
  else:
    raise ValueError('Unsupported termination strategy: {}'.format(termination))

class TerminationAgent(Agent):
  def __init__(self, agent, strategy=None):
    Agent.__init__(self)
    self.agent = agent
    self.strategy = strategy if strategy is not None else TerminationStrategy()
  def select_move(self, gs):
    if self.strategy.should_pass(gs):
      return Move.pass_turn()
    elif self.strategy.should_resign(gs):
      return Move.resign()
    else:
      return self.agent.select_move(gs)
