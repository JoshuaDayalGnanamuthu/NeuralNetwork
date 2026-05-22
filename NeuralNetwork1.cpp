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

      
