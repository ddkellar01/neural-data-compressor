import unittest
import numpy as np
from src.entropy.arithmetic_coder import SimpleRansCoder

class TestEntropyCoder(unittest.TestCase):
    def setUp(self):
        self.coder = SimpleRansCoder(state_bits=32)

    def test_bitstream_generation_and_length(self):
        symbols = np.array([0, 1, 3, 2, 0, 1, 1, 0], dtype=np.int32)
        probabilities = np.array([0.4, 0.3, 0.1, 0.2, 0.4, 0.3, 0.3, 0.4], dtype=np.float32)
        
        bitstream = self.coder.encode_symbols(symbols, probabilities)
        self.assertIsInstance(bitstream, bytes)
        self.assertGreater(len(bitstream), 0, "Generated bitstream payload is empty.")

if __name__ == "__main__":
    unittest.main()
