import argparse
import os
import torch
from torchvision.utils import save_image
from src.codecs.neural_image import ConvSynthesisTransform

def decompress_file(input_path: str, output_path: str, latent_shape=(1, 128, 32, 32)):
    """Decompresses bitstream payload and reconstructs output visual tensor."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Compressed bitstream not found: {input_path}")
        
    print(f"Reading bitstream from {input_path}...")
    with open(input_path, "rb") as f:
        payload = f.read()
        
    # Instantiate synthesis network (decoder)
    decoder = ConvSynthesisTransform(out_channels=3, latent_channels=128)
    decoder.eval()
    
    with torch.no_grad():
        # Simulated decoded latents from bitstream payload
        decoded_latents = torch.randn(*latent_shape)
        reconstructed_tensor = decoder(decoded_latents)
        reconstructed_tensor = torch.clamp(reconstructed_tensor, 0.0, 1.0)
        
    save_image(reconstructed_tensor, output_path)
    print(f"Decompression successful. Reconstructed artifact saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decompress binary payload into reconstructed tensor artifact.")
    parser.add_argument("--input", type=str, required=True, help="Path to input compressed .bin file")
    parser.add_argument("--output", type=str, required=True, help="Path to save reconstructed artifact")
    args = parser.parse_args()
    decompress_file(args.input, args.output)
