#include "NeuralNetwork.hpp"

/**
 */
void NerualNetwork::sigmoid(Matrix &Object) {
  for ( size_t i {}; i < Object.row; ++i ) {
    for ( size_t j {}; j < Object.column; ++j ) {
      Object[i][j] = 1.0 / (1.0 + std::exp(-Object[i][j]));
    }
  }
}

/**
 */
void NeuralNetwork::relu(Matrix &Object) {
  for ( size_t i {}; i < Object.row; ++i ) {
    for ( size_t j {}; j < Object.column; ++j) {
      Object[i][j] = (Object[i][j] > 0.0) ? Object[i][j] : 0.0;
    }
  }
}

/**
 */
void NeuralNetwork::leakyrelu(Matrix &Object, double alpha = 0.01) {
  for ( size_t i {}; i < Object.row; ++i ) {
    for ( size_t j {}; j < Object.column; ++j ) {
      Object[i][j] = (Object[i][j] > 0.0) ? Object[i][j] : alpha * Object[i][j];
    }
  }
}

/**
 */
void NeuralNetwork::initialize_parameters() {
  for ( size_t i {1}; i < this->layers.size; ++i ) {
    this->Weights["W" + std::to_string(i)] = Matrix(row = this->layers[i], column =  this->layers[i - 1]);
    this->basis["b" + std::to_string(i)] = Matrix(row = this->layers[i], column = 1, value = 0);
  }
}
