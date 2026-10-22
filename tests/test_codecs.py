import unittest
import torch
from src.codecs.neural_image import ConvAnalysisTransform, ConvSynthesisTransform
from src.codecs.weight_quant import UniformQuantizer

class TestCodecTransforms(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.in_channels = 3
        self.height, self.width = 128, 128
        self.encoder = ConvAnalysisTransform(in_channels=self.in_channels, latent_channels=128)
        self.decoder = ConvSynthesisTransform(out_channels=self.in_channels, latent_channels=128)

    def test_spatial_reconstruction_shape(self):
        x = torch.randn(self.batch_size, self.in_channels, self.height, self.width)
        latents = self.encoder(x)
        reconstruction = self.decoder(latents)
        self.assertEqual(x.shape, reconstruction.shape, "Reconstructed spatial shape does not match input.")

    def test_uniform_quantizer_roundtrip(self):
        quantizer = UniformQuantizer(num_bits=4)
        weights = torch.randn(64, 64)
        q_weights, scale = quantizer.quantize(weights)
        dequantized = quantizer.dequantize(q_weights, scale)
        self.assertEqual(q_weights.dtype, torch.int8)
        self.assertEqual(weights.shape, dequantized.shape)

if __name__ == "__main__":
    unittest.main()
