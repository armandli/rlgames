import unittest

from rlgames.goscore import AreaScore
from rlgames.goboard import Board
from rlgames.common_types import Player, Point

class ScoringTest(unittest.TestCase):
    def test_scoring(self):
        # .w.ww
        # wwww.
        # bbbww
        # .bbbb
        # .b.b.
        board = Board(5)
        board.place_stone(Player.black, Point(1, 2))
        board.place_stone(Player.black, Point(1, 4))
        board.place_stone(Player.black, Point(2, 2))
        board.place_stone(Player.black, Point(2, 3))
        board.place_stone(Player.black, Point(2, 4))
        board.place_stone(Player.black, Point(2, 5))
        board.place_stone(Player.black, Point(3, 1))
        board.place_stone(Player.black, Point(3, 2))
        board.place_stone(Player.black, Point(3, 3))
        board.place_stone(Player.white, Point(3, 4))
        board.place_stone(Player.white, Point(3, 5))
        board.place_stone(Player.white, Point(4, 1))
        board.place_stone(Player.white, Point(4, 2))
        board.place_stone(Player.white, Point(4, 3))
        board.place_stone(Player.white, Point(4, 4))
        board.place_stone(Player.white, Point(5, 2))
        board.place_stone(Player.white, Point(5, 4))
        board.place_stone(Player.white, Point(5, 5))
        scorer = AreaScore(board)
        self.assertEqual(Player.white, scorer.winner())
        self.assertEqual(9, scorer.bp)
        self.assertEqual(4, scorer.bt)
        self.assertEqual(9, scorer.wp)
        self.assertEqual(3, scorer.wt)
        self.assertEqual(0, scorer.dames)
