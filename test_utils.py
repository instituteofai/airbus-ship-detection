import unittest
from utility import rle_decode, rle_encode
import numpy as np


class TestRLEMethods(unittest.TestCase):

  def test_sanity(self):
    self.assertEqual(True, True)

  def test_encoder_1(self):

    img = np.array([[0, 0, 0, 0],
                    [0, 1, 1, 0],
                    [0, 1, 1, 0],
                    [0, 0, 0, 0]])
    expected_mask = '6 2 10 2'

    result = rle_encode(img)
    self.assertEqual(
        result,
        expected_mask,
        f'Expected: {expected_mask}, Received: {result}'
    )

  def test_encoder_2(self):

    img = np.array([[0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]])
    expected_mask = '9 5 18 4 27 3 36 2 45 1'

    result = rle_encode(img)
    self.assertEqual(
        result,
        expected_mask,
        f'Expected: {expected_mask}, Received: {result}'
    )

  def test_decoder_1(self):

    expected_img = np.array([ [1, 0, 0, 0],
                              [0, 1, 1, 0],
                              [0, 1, 1, 0],
                              [0, 0, 0, 0]])
    mask = '1 1 6 2 10 2'

    result = rle_decode(mask, (4, 4))
    self.assertEqual(
        np.array_equal(result, expected_img),
        True,
        f'Expected:\n{expected_img}\n\nReceived:\n{result}'
    )

  def test_decoder_2(self):

    expected_img = np.array([ [0, 1, 0, 0, 0, 0, 0],
                              [0, 1, 1, 0, 0, 0, 0],
                              [0, 1, 1, 1, 0, 0, 0],
                              [0, 1, 1, 1, 1, 0, 0],
                              [0, 1, 1, 1, 1, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0]])
    mask = '9 5 18 4 27 3 36 2 45 1'

    result = rle_decode(mask, (8, 7))
    self.assertEqual(
        np.array_equal(result, expected_img),
        True,
        f'Expected:\n{expected_img}\n\nReceived:\n{result}'
    )

  def test_decoder_3(self):

    expected_img = np.array([ [0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 1, 0, 0, 0, 0, 0],
                              [0, 1, 1, 1, 0, 0, 0, 0],
                              [0, 1, 1, 1, 1, 0, 0, 0],
                              [0, 1, 1, 1, 1, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0]])
    mask = '9 5 18 4 27 3 36 2 45 1'

    result = rle_decode(mask, (8, 8))
    self.assertEqual(
        np.array_equal(result, expected_img),
        True,
        f'Expected:\n{expected_img}\n\nReceived:\n{result}'
    )
