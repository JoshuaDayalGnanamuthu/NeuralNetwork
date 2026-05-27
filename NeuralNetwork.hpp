#include <unordered_map>
#include <string>
#include "Matrix.hpp"

class NeuralNetwork {
  /**
   *Initialize a Neural Network class with a customizeable architecture.
   * Params:
   *
   */
public:
  NeuralNetwork(std::vector<int> hidden_layers= {2, 3, 4, 3, 1}; acitivation = "relu", final_Activation = "sigmoid") {};
  
private:
  int input_size;
  std::vector<int> layers;
  std::string activation;
  std::function<void(Matrix)> final_activation;
  
  std::unordered_map<std::string, Matrix> weights;
  std::unordered_map<std::string, Matrix> baises;
  std::unordered_map<std::string, Matrix> Nodes;
  std::unordered_map<std::string, std::vector<std::function<void(Matrix)>>> activations = {
    "sigmoid": {sigmoid, sigmoid_derivative},
    "relu": {relu, relu_derivative},
    "leakyrelu": {leakyrelue, leakyrelu_derivative}
  };
  
  void sigmoid(Matrix &Object);
  void relu(Matrix &Object);
  void leakyrelu(Matrix &Object, double aplha = 0.01);

  void sigmoid_derivate(Matrix &Object);
  void relu_derivative(Matrix &Object);
  void leakyrelu_derivatvie(Matrix &Object);

  void initialize_parameters();
  
  void setdefault();
  void feedforward();
  void backpropogation();
  
  void analysis();
  double cost(Matrix &Node, Matrix &Y);

  void train(Matrix &X, Matrix &Y, int epochs = 1000, double learning_rate = 0.01, std::optional<int> batch_size, int print_interval = 100, double learning_rate_decay = 0.95, double decay_interval = 100, std::optional<int> early_stopping_patience, std::optional<Matrix> &validation_data, std::optional<Matrix> &class_Weights);
 
 
 
};
