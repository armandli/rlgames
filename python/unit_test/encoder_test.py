import unittest

from rlgames.encoders.base import get_encoder_by_name
from rlgames.common_types import Point
from rlgames.game_base import Move
from rlgames.goboard import GameState

class OnePlaneEncoderTest(unittest.TestCase):
  def test_create(self):
    encoder = get_encoder_by_name('oneplane', 9)
    self.assertTrue(encoder is not None)
  def test_encode_point(self):
    encoder = get_encoder_by_name('oneplane', 9)
    pt = Point(2, 2)
    idx = encoder.encode_point(pt)
    self.assertEqual(10, idx)
  def test_decode_index(self):
    encoder = get_encoder_by_name('oneplane', 9)
    pt = encoder.decode_point_index(16)
    self.assertEqual(Point(2, 8), pt)
  def test_num_points(self):
    encoder = get_encoder_by_name('oneplane', 9)
    self.assertEqual(81, encoder.num_points())
  def test_shape(self):
    encoder = get_encoder_by_name('oneplane', 9)
    shape = encoder.shape()
    self.assertEqual(1, shape[0])
    self.assertEqual(9, shape[1])
    self.assertEqual(9, shape[2])
  def test_encode(self):
    encoder = get_encoder_by_name('oneplane', 9)
    gs = GameState.new_game(9)
    gs = gs.apply_move(Move.play(Point(5, 5)))
    gs = gs.apply_move(Move.play(Point(4, 5)))
    code = encoder.encode(gs)
    self.assertEqual(1, code[0][4][4])
    self.assertEqual(-1, code[0][3][4])

class SevenPlaneEncoderTest(unittest.TestCase):
  def test_create(self):
    encoder = get_encoder_by_name('sevenplane', 9)
    self.assertTrue(encoder is not None)
  def test_encode_point(self):
    encoder = get_encoder_by_name('sevenplane', 9)
    pt = Point(2, 2)
    idx = encoder.encode_point(pt)
    self.assertEqual(10, idx)
  def test_decode_index(self):
    encoder = get_encoder_by_name('sevenplane', 9)
    pt = encoder.decode_point_index(16)
    self.assertEqual(Point(2, 8), pt)
  def test_num_points(self):
    encoder = get_encoder_by_name('sevenplane', 9)
    self.assertEqual(81, encoder.num_points())
  def test_shape(self):
    encoder = get_encoder_by_name('sevenplane', 9)
    shape = encoder.shape()
    self.assertEqual(7, shape[0])
    self.assertEqual(9, shape[1])
    self.assertEqual(9, shape[2])
  def test_encode(self):
    encoder = get_encoder_by_name('sevenplane', 9)
    gs = GameState.new_game(9)
    gs = gs.apply_move(Move.play(Point(2, 7)))
    gs = gs.apply_move(Move.play(Point(7, 2)))
    gs = gs.apply_move(Move.play(Point(3, 6)))
    gs = gs.apply_move(Move.play(Point(6, 3)))
    gs = gs.apply_move(Move.play(Point(3, 7)))
    gs = gs.apply_move(Move.play(Point(2, 6)))
    gs = gs.apply_move(Move.play(Point(2, 5)))
    code = encoder.encode(gs)
    self.assertEqual(1., code[0][1][5])

class ElevenPlaneEncoderTest(unittest.TestCase):
  def test_create(self):
    encoder = get_encoder_by_name('elevenplane', 9)
    self.assertTrue(encoder is not None)
  def test_encode_point(self):
    encoder = get_encoder_by_name('elevenplane', 9)
    pt = Point(2, 2)
    idx = encoder.encode_point(pt)
    self.assertEqual(10, idx)
  def test_decode_index(self):
    encoder = get_encoder_by_name('elevenplane', 9)
    pt = encoder.decode_point_index(16)
    self.assertEqual(Point(2, 8), pt)
  def test_num_points(self):
    encoder = get_encoder_by_name('elevenplane', 9)
    self.assertEqual(81, encoder.num_points())
  def test_shape(self):
    encoder = get_encoder_by_name('elevenplane', 9)
    shape = encoder.shape()
    self.assertEqual(11, shape[0])
    self.assertEqual(9, shape[1])
    self.assertEqual(9, shape[2])
  def test_encode(self):
    encoder = get_encoder_by_name('elevenplane', 9)
    gs = GameState.new_game(9)
    gs = gs.apply_move(Move.play(Point(2, 7)))
    gs = gs.apply_move(Move.play(Point(7, 2)))
    gs = gs.apply_move(Move.play(Point(3, 6)))
    gs = gs.apply_move(Move.play(Point(6, 3)))
    gs = gs.apply_move(Move.play(Point(3, 7)))
    gs = gs.apply_move(Move.play(Point(2, 6)))
    gs = gs.apply_move(Move.play(Point(2, 5)))
    code = encoder.encode(gs)
    self.assertEqual(1., code[4][1][5])
