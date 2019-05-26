import copy
from gotypes import Player, Point
import zobrist

# There are 3 types of moves player can make: place piece, pass, or resign
# a good bot should know how to resign
class Move(object):
  def __init__(self, point=None, is_pass=False, is_resign=False):
    assert (point is not None) ^ is_pass ^ is_resign
    self.pt = point
    self.is_play = self.pt is not None
    self.is_pass = is_pass
    self.is_resign = is_resign

  @classmethod
  def play(cls, point):
    return cls(point)

  @classmethod
  def pass_turn(cls):
    return cls(is_pass = True)

  @classmethod
  def resign(cls):
    return cls(is_resign = True)

# Tracks groups of stones of the same color to speed up checking for liberties
class GoString(object):
  def __init__(self, color, stones, liberties):
    self.color = color
    self.stones = frozenset(stones)
    self.liberties = frozenset(liberties)

  def without_liberty(self, pt):
    new_liberties = self.liberties - set([pt])
    return GoString(self.color, self.stones, new_liberties)

  def with_liberty(self, pt):
    new_liberties = self.liberties | set([pt])
    return GoString(self.color, self.stones, new_liberties)

  def merge_with(self, gostring):
    assert gostring.color == self.color
    combined_stones = self.stones | gostring.stones
    return GoString(self.color, combined_stones, (self.liberties | gostring.liberties) - combined_stones)

  @property
  def num_liberties(self):
    return len(self.liberties)

  def __eq__(self, other):
    return isinstance(other, GoString) and \
           self.color == other.color and \
           self.stones == other.stones and \
           self.liberties == other.liberties

class Board(object):
  def __init__(self, num_rows, num_cols):
    self.nrows = num_rows
    self.ncols = num_cols
    self.grid = dict()
    self.hash = zobrist.EMPTY_BOARD

  def zobrist_hash(self):
    return self.hash

  def is_on_grid(self, pt):
    return 1 <= pt.r <= self.nrows and 1 <= pt.c <= self.ncols
  
  def get(self, pt):
    string = self.grid.get(pt)
    if string is None:
      return None
    else:
      return string.color

  def get_go_string(self, pt):
    return self.grid.get(pt)

  def remove_string_(self, string):
    for pt in string.stones:
      for neighbour in pt.neighbours():
        neighbour_string = self.grid.get(neighbour)
        if neighbour_string is None:
          continue
        if neighbour_string is not string:
          self.replace_string_(neighbour_string.with_liberty(pt))
      self.grid[pt] = None
      self.hash ^= zobrist.HASH_CODE[pt, string.color]

  def replace_string_(self, string):
    for pt in string.stones:
      self.grid[pt] = string

  def place_stone(self, player, pt):
    assert self.is_on_grid(pt)
    assert self.grid.get(pt) is None
    #small set, use list
    adj_same_color = []
    adj_oppo_color = []
    liberties = []
    for neighbour in pt.neighbours():
      if not self.is_on_grid(neighbour):
        continue
      neighbour_string = self.grid.get(neighbour)
      if neighbour_string is None:
        liberties.append(neighbour)
      elif neighbour_string.color == player:
        if neighbour_string not in adj_same_color:
          adj_same_color.append(neighbour_string)
      else:
        if neighbour_string not in adj_oppo_color:
          adj_oppo_color.append(neighbour_string)
    new_string = GoString(player, [pt], liberties)
    for string in adj_same_color:
      new_string = new_string.merge_with(string)
    for pnt in new_string.stones:
      self.grid[pnt] = new_string
    self.hash ^= zobrist.HASH_CODE[pt, player]
    #self capture is considered illegal move, we don't check
    #this is consistent in most rule sets
    #it is also important to remove opponent's captured string
    #before removing your own, otherwise it breaks correct
    #play
    for string in adj_oppo_color:
      replacement = string.without_liberty(pt)
      if replacement.num_liberties:
        self.replace_string_(replacement)
      else:
        self.remove_string_(replacement)
    for string in adj_oppo_color:
      if string.num_liberties == 0:
        self.remove_string_(string)

class GameState(object):
  def __init__(self, board, next_player, previous, move):
    self.board = board
    self.nplayer = next_player
    self.prev = previous
    if self.prev is None:
      self.prevs = frozenset()
    else:
      self.prevs = frozenset(previous.prevs | {(previous.nplayer, previous.board.zobrist_hash())})
    self.pmove = move
  
  @property
  def situation(self):
    return (self.nplayer, self.board)

  def apply_move(self, move):
    if move.is_play:
      next_board = copy.deepcopy(self.board)
      next_board.place_stone(self.nplayer, move.pt)
    else:
      next_board = self.board
    return GameState(next_board, self.nplayer.other, self, move)

  def is_over(self):
    if self.pmove is None: #if it is the first board
      return False
    if self.pmove.is_resign:
      return True
    if self.prev.pmove is None: #second board
      return False
    return self.pmove.is_pass and self.prev.pmove.is_pass

  def is_move_self_capture(self, player, move):
    if not move.is_play:
      return False
    #not very efficient here, should avoid deep copy
    next_board = copy.deepcopy(self.board)
    next_board.place_stone(player, move.pt)
    new_string = next_board.get_go_string(move.pt)
    return new_string.num_liberties == 0

  #go rule where you cannot make a move that looks exactly
  #the same as previous state
  def does_move_violate_ko(self, player, move):
    if not move.is_play:
      return False
    #not very efficient here, should avoid deep copy
    next_board = copy.deepcopy(self.board)
    next_board.place_stone(player, move.pt)
    next_situation = (player.other, next_board.zobrist_hash())
    return next_situation in self.prevs

  def is_valid_move(self, move):
    if self.is_over():
      return False
    if move.is_pass or move.is_resign:
      return True
    #TODO: bug, it is legal to make self capture move but remove
    #opponent string first
    return (
      self.board.get(move.pt) is None and
      not self.is_move_self_capture(self.nplayer, move) and
      not self.does_move_violate_ko(self.nplayer, move))

  def legal_moves(self):
    if self.is_over():
      return list()
    ret = [Move.pass_turn, Move.resign]
    for ri in range(1, self.board.nrows + 1):
      for ci in range(1, self.board.ncols + 1):
        m = Move.play(Point(ri, ci))
        if self.is_valid_move(m):
          ret.append(m)
    return ret

  def winner():
    assert self.is_over() #need to call is_over first
    if self.pmove == Move.resign:
      return self.nplayer
    #1: remove gostrings that have less than 2 eyes (liberties that are also eyes)
    #2: count number of pieces per player
    #3: count number of territories per player, a territory is a empty location fully surrounded by one player
    #TODO: count the pieces according to international rule. winner is the one with bigger count. note white has additional base points

  @classmethod
  def new_game(cls, board_size):
    if isinstance(board_size, int):
      board = Board(board_size, board_size)
      return cls(board, Player.black, None, None)