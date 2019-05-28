import copy
from rlgames.common_types import Player, Point
from rlgames.game_base import Move, BoardBase, GameStateBase

class Board(BoardBase):
  def __init__(self, size):
    self.sz = size
    self.grid = dict()

  def is_on_grid(self, pt):
    return 1 <= pt.r <= self.sz and 1 <= pt.c <= self.sz

  def get(self, pt):
    return self.grid.get(pt)

  def place_stone(self, player, pt):
    assert self.is_on_grid(pt)
    assert self.grid.get(pt) is None
    self.grid[pt] = player

class GameState(GameStateBase):
  def __init__(self, board, next_player, previous, move):
    self.board = board
    self.nplayer = next_player
    self.pmove = move
    #no need to keep the previous states

  def apply_move(self, move):
    assert self.is_over() == False
    if move.is_play:
      new_board = copy.deepcopy(self.board)
      new_board.place_stone(self.nplayer, move.pt)
      return GameState(new_board, self.nplayer.other, self, move)
    else:
      return GameState(self.board, self.nplayer.other, self, move)

  def is_connected_(self):
    assert self.pmove.is_play
    pt = self.pmove.pt
    player = self.nplayer.other
    if (
      sum([self.board.get(Point(ri, pt.c)) == player for ri in range(1, self.board.sz + 1)]) == self.board.sz or
      sum([self.board.get(Point(pt.r, ci)) == player for ci in range(1, self.board.sz + 1)]) == self.board.sz or
      sum([self.board.get(Point(i, i)) == player for i in range(1, self.board.sz + 1)]) == self.board.sz or
      sum([self.board.get(Point(i, self.board.sz + 1 - i)) == player for i in range(1, self.board.sz + 1)]) == self.board.sz):
      return True
    return False

  def empty_positions_(self):
    ret = list()
    for ri in range(1, self.board.sz + 1):
      for ci in range(1, self.board.sz + 1):
        pt = Point(ri, ci)
        if self.board.get(pt) is None:
          ret.append(pt)
    return ret

  def is_over(self):
    if self.pmove is None:
      return False
    if self.pmove.is_resign or self.pmove.is_pass:
      return True
    if self.is_connected_():
      return True
    return len(self.empty_positions_()) == 0

  def is_valid_move(self, move):
    return self.board.get(move.pt) is None

  def legal_moves(self):
    if self.is_over():
      return list()
    ret = [Move.play(pt) for pt in self.empty_positions_()]
    ret.append(Move.pass_turn())
    return ret

  def winner(self):
    if self.pmove is None:
      return None
    if self.pmove.is_resign or self.pmove.is_pass:
      return self.nplayer
    if self.is_connected_():
      return self.nplayer.other
    return None

  @classmethod
  def new_game(cls, board_size):
    board = Board(board_size)
    return cls(board, Player.black, None, None)
