import enum
from collections import namedtuple

class Player(enum.Enum):
  black = 1
  white = 2

  @property
  def other(self):
    return Player.black if self == Player.white else Player.white

class Point(namedtuple('Point', 'r c')):
  def neighbours(self):
    return [
      Point(self.r - 1, self.c),
      Point(self.r + 1, self.c),
      Point(self.r, self.c - 1),
      Point(self.r, self.c + 1)
    ]

  def corners(self):
    return [
      Point(self.r - 1, self.c - 1),
      Point(self.r - 1, self.c + 1),
      Point(self.r + 1, self.c - 1),
      Point(self.r + 1, self.c + 1),
    ]

  def __deepcopy__(self, memodict={}):
    # very immutable
    return self
