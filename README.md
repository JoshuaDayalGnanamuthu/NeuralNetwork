# Modular Neural Network (C++)

A compact C++ implementation of a customizable multi-layer perceptron (MLP) for binary classification. The architecture, activations, and training loop are built from scratch with modular utility functions, allowing easy experimentation with layer sizes and learning hyperparameters.

## Project structure

- `NeuralNetwork.cpp` - end-to-end neural network implementation with dataset, training loop, and prediction output.
- `ModularNeuralNetwork.cpp` - equivalent modular version with shared utility routines.

## Model architecture

- Input layer: 2 features (`X` for weight and height)
- Hidden layers: 3, 4, 3 neurons
- Output layer: 1 neuron
- Activation functions:
  - hidden: Leaky ReLU + layer normalization
  - output: Sigmoid
- Cost: binary cross-entropy

## Key features

- Custom matrix operations: `multiply`, `add`, `subtract`, `transpose`, `broadcast`
- Weight initialization: He normal
- Normalization: standard (per-feature mean/std scaling)
- Backpropagation with gradients for weights/biases
- Predict probabilities and compare with ground truth

## How to build & run

From project folder:

```bash
# compile
c++ -std=c++17 NeuralNetwork.cpp -o neural
# or
c++ -std=c++17 ModularNeuralNetwork.cpp -o modular_neural

# run
./neural
# or
./modular_neural
```

## Hyperparameters (hardcoded)

- `layers` = `{2,3,4,3,1}`
- `epochs` = `1_000_000`
- `learning_rate` = `0.01`
- `batch` = all samples (10 points) 

## Sample dataset

Binary labels for 10 samples (0/1):
- `X`: 2 x 10 matrix (weight, height)
- `Y`: 1 x 10 label vector

## Notes / possible improvements

- add command-line options for model config and epochs
- add mini-batch or stochastic gradient descent for large datasets
- extract data loading from CSV (currently uses hard-coded data)
- avoid global state (`Weights`, `Baises`, `Nodes`) in production code
- add evaluation metrics (accuracy, precision, recall)

## Output examples

The program prints initial cost and progress every `10000` epochs, and final raw predicted probabilities against known `Y` labels.

---

Built for learning and exploration; not yet optimized for production use.
