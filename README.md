# Neural Data & Model Compression Engine (`neural-data-compressor`)

[![Continuous Integration](https://github.com/ddkeller01/neural-data-compressor/actions/workflows/ci.yml/badge.svg)](https://github.com/ddkeller01/neural-data-compressor/actions)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🧰 Tech Stack Badges & Architecture

### 🛠️ Tech Stack Badges

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.22%2B-013243?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-1.8%2B-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![CUDA](https://img.shields.io/badge/NVIDIA_CUDA-12.1-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-24.0%2B-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

### 💻 Core Programming Languages & Core Systems
* **Python (3.9+):** Drives the primary system engines, deep learning layers, data pipelines, optimization loops, and CLI entry points (`compress.py`, `decompress.py`).
* **Deep Learning Frameworks:** `PyTorch` (tensor operations, autograd, and GPU execution) and `Torchvision` (image transformations and tensor spatial operations).
* **Scientific Computing & IO:** `NumPy` & `SciPy` (mathematical arrays, probability distributions) and `Pillow` (image decoding/encoding).

---

### 🧰 Architecture Overview

```text
+-----------------------------------------------------------------------------------+
|                            Neural Data Compressor Engine                          |
+-----------------------------------------------------------------------------------+
                                          |
           +------------------------------+------------------------------+
           |                                                             |
           v                                                             v
+-------------------------------+                             +---------------------+
|     Perceptual Data Codec     |                             | Weight Quantization |
|      (Image & Signals)        |                             |  (LLM Parameters)   |
+-------------------------------+                             +---------------------+
| • ConvAnalysisTransform       |                             | • Uniform Quantizer |
| • GDN Normalization           |                             | • Activation Scales |
| • Gaussian Entropy Model      |                             | • FP16 -> INT4/INT3 |
+---------------+---------------+                             +----------+----------+
                |                                                        |
                v                                                        v
+-----------------------------------------------------------------------------------+
|                        Lossless Entropy Coding Engine                             |
|       (Register Asymmetric Numeral Systems / rANS with Overflow Protection)       |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        Dense Output Binary Stream (.bin)                          |
+-----------------------------------------------------------------------------------+

A unified, high-performance open-source engine bridging two major paradigms of artificial intelligence compression: **Neural Signal Codecs** (compressing images, audio, and data streams into compact latent bitstreams via Rate-Distortion optimization) and **Model Parameter Quantization** (compressing massive Large Language Model weights down to low-bit precision without losing intelligence).

---

## 🌟 Core Features

* **Perceptual Neural Codecs:** End-to-end learned image/signal compression using Generalized Divisive Normalization (`GDN`) and Rate-Distortion optimization.
* **Overflow-Protected rANS Coder:** Implements Asymmetric Numeral Systems (Register ANS) arithmetic coding with arbitrary-precision integer tracking for robust binary stream serialization.
* **LLM Weight Quantization & Calibration:** Activation-aware weight quantization routines reducing model memory footprints from FP16 down to INT4/INT3.
* **Modular Pipeline Architecture:** Fully decoupled components for analysis transforms, entropy models, probability priors, and loss calculations.
* **Containerized Workflows:** Ready-to-use Docker and Docker Compose configurations supporting GPU-accelerated execution.

---

## 📁 Project Structure Tree

```text
neural-data-compressor/
├── .github/
│   └── workflows/
│       ├── ci.yml                   # Automated unit testing and linting
│       └── release.yml              # Build and publish pipelines
├── configs/
│   ├── default_config.yaml          # Global hyperparameters (learning rate, lambda)
│   ├── image_codec.yaml             # Settings for learned image/video compression
│   └── model_quantization.yaml      # Settings for LLM/weight quantization (AWQ/GPTQ)
├── data/
│   ├── dataset_loader.py            # High-throughput data streaming pipelines
│   └── preprocessor.py              # Normalization, tokenization, and patching utilities
├── src/
│   ├── __init__.py
│   ├── codecs/
│   │   ├── __init__.py
│   │   ├── calibration.py           # Activation-aware scale calibration
│   │   ├── neural_image.py          # Encoder/Decoder transforms with GDN layers
│   │   └── weight_quant.py          # Weight quantization & calibration algorithms
│   ├── entropy/
│   │   ├── __init__.py
│   │   ├── arithmetic_coder.py      # rANS arithmetic coder with overflow protection
│   │   └── probability_model.py     # Context models estimating symbol probabilities
│   ├── losses/
│   │   ├── __init__.py
│   │   └── rate_distortion.py       # Joint loss math balancing file size vs. fidelity
│   └── utils/
│       ├── __init__.py
│       └── metrics.py               # BPP, PSNR, and Compression calculations
├── scripts/
│   ├── train.py                     # Main training loop for neural codecs
│   ├── compress.py                  # CLI script to compress raw files into custom bitstreams
│   └── decompress.py                # CLI script to reconstruct files from compressed bitstreams
├── tests/
│   ├── test_codecs.py               # Unit tests for forward/inverse codec consistency
│   └── test_entropy.py              # Verification tests for lossless entropy round-trips
├── Dockerfile                       # Container definition with GPU acceleration support
├── docker-compose.yml               # Multi-container local testing setup
├── pyproject.toml                   # Project metadata and dependency management
├── setup.py                         # Fallback build hook for editable installations
├── README.md                        # Architecture documentation & setup instructions
└── requirements.txt                 # Python dependencies (PyTorch, NumPy, SciPy)
