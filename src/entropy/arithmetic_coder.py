import numpy as np

class SimpleRansCoder:
    """rANS (Register Asymmetric Numeral Systems) codec wrapper with overflow protection."""
    def __init__(self, state_bits=32):
        self.state_bits = state_bits
        self.l_state = 1 << (state_bits - 1)

    def encode_symbols(self, symbols: np.ndarray, probabilities: np.ndarray) -> bytes:
        # Use Python native integer for state to avoid fixed-width int32 overflow
        state = int(self.l_state)
        bitstream = bytearray()
        
        for sym, prob in zip(reversed(symbols), reversed(probabilities)):
            freq = max(1, int(prob * 4096))
            # Safe Python arbitrary precision arithmetic
            state = int(((state // freq) << 12) + (state % freq) + int(sym))
            
            while state >= (1 << 24):
                bitstream.append(int(state & 0xFF))
                state >>= 8
                
        bitstream.extend(int(state).to_bytes(4, 'little', signed=False))
        return bytes(bitstream)
