from rlgames.common_types import Player, Point

#TODO: test

#scoring using area score rule: player pieces + territory + komi
class AreaScore(object):
  def __init__(self, board):
    self.board = board
    self.labels = dict()
    self.remove_dead_strings_()
    self.create_territory_labels_()
    (self.bp, self.bt, self.wp, self.wt, self.dames) = self.score_()
    self.komi = 7.5

  def winner(self):
    if self.bp + self.bt > self.wp + self.wt + self.komi:
      return Player.black
    else:
      return Player.white

  def winning_margin(self):
    return self.bp - self.wp + self.bt - self.wt - self.komi

  def recursive_territory_labeling_(self, start_pt, points, boundary):
    for neighbour in start_pt.neighbours():
      if not self.board.is_on_grid(neighbour):
        continue
      color = self.board.get(neighbour)
      if neighbour not in points and color is None:
        points.add(neighbour)
        (tpoints, tboundary) = self.recursive_territory_labeling_(neighbour, points, boundary)
        points |= tpoints
        boundary |= tboundary
      elif color is not None:
        boundary.add(color)
    return (points, boundary)

  def create_territory_labels_(self):
    self.labels.clear()
    for ri in range(1, self.board.sz + 1):
      for ci in range(1, self.board.sz + 1):
        pt = Point(ri, ci)
        if self.board.get(pt) is None and pt not in self.labels:
          (tpoints, boundary) = self.recursive_territory_labeling_(pt, {pt}, set())
          if len(boundary) == 1:
            color = boundary.pop()
            for pti in tpoints:
              self.labels[pti] = color
          elif len(boundary) == 2:
            for pti in tpoints:
              self.labels[pti] = 'dame'

  def remove_dead_strings_(self):
    self.create_territory_labels_()
    for ri in range(1, self.board.sz + 1):
      for ci in range(1, self.board.sz + 1):
        string = self.board.get_go_string_(Point(ri, ci))
        if string is None:
          continue
        eye_count = 0
        for liberty in string.liberties:
          if self.labels.get(liberty) == string.color:
            eye_count += 1
        if eye_count < 2:
          self.board.remove_string_(string)

  def score_(self):
    wp = 0
    wt = 0
    bp = 0
    bt = 0
    dm = 0
    for ri in range(1, self.board.sz + 1):
      for ci in range(1, self.board.sz + 1):
        pt = Point(ri, ci)
        color = self.board.get(pt)
        if color is None:
          tcolor = self.labels.get(pt)
          if tcolor == Player.black:
            bt += 1
          elif tcolor == Player.white:
            wt += 1
          else:
            dm += 1
        elif color == Player.black:
          bp += 1
        else:
          wp += 1
    return (bp, bt, wp, wt, dm)
