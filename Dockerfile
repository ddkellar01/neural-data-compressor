[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "neural-data-compressor"
version = "0.1.0"
description = "A unified neural data compression engine and LLM parameter quantization framework"
readme = "README.md"
requires-python = ">=3.9"
license = {text = "MIT"}
authors = [
    {name = "darnellwashingtonjr94-art"}
]
dependencies = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "numpy>=1.22.0",
    "scipy>=1.8.0",
    "pyyaml>=6.0",
    "pillow>=9.0.0"
]

[project.scripts]
neural-compress = "scripts.compress:compress_file"
neural-decompress = "scripts.decompress:decompress_file"
