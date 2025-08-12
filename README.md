# Transformer from Scratch

This repository contains a minimal, from-scratch implementation of the **Transformer** architecture, inspired by the original paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762).

## Features
- **Scaled Dot-Product Attention** implementation
- **Multi-Head Attention** with residual connections & LayerNorm
- **Position-wise Feed-Forward** layers
- **Sinusoidal Positional Encoding**
- **Stackable Encoder layers** for deeper models
- Example training loop with dummy data
- Easily extendable to Decoder and full Seq2Seq tasks

## File Structure
transformer-from-scratch/
│
├── transformer.py # Core Transformer encoder implementation
├── train_demo.py # Simple training demo
└── README.md # Project description

## Installation
```bash
git clone https://github.com/<your-username>/transformer-from-scratch.git
cd transformer-from-scratch
pip install torch matplotlib

## Usage
Run the demo training loop:
python train_demo.py

## Example Output
Epoch 1, Loss: 4.5912
Epoch 2, Loss: 4.2128
...

## Author
randarura

## License
This project is licensed under the MIT License – see the LICENSE file for details.
