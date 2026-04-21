# synth-perception-numpy

A feedforward neural network built from scratch using only NumPy for plant species classification. Built as a training exercise following the Neural Network and Deep Learning courses by Andrew Ng, and to be used as the perception foundation for the SynthField synthetic data pipeline targeting agricultural robotics and Isaac Sim deployment.

## What this is
This is not a framework wrapper. Every component is implemented manually:
- Forward pass (linear + ReLU/sigmoid activations)
- Cross entropy cost function
- Backpropagation
- Gradient descent parameter updates

## Why NumPy
Understanding what happens underneath frameworks like TensorFlow and PyTorch matters. This implementation proves the math works before abstracting it away.

## Results
- 2 plant classes: tomato, corn
- 10 training images
- 1000 iterations
- Training accuracy: 100%
- Cost reduced from 0.691 to 0.005

## Part of SynthField Labs
This perception module is the foundation of a larger synthetic data pipeline. Training data will be generated using Houdini procedural renders for deployment in Isaac Sim agricultural robotics applications.

## Usage
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Folder Structure

data/
├── tomato/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── img3.jpg
└── corn/
    ├── img1.jpg
    ├── img2.jpg
    └── img3.jpg

## Stack
- Python 3
- NumPy
- Pillow