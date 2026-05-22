#include <unordered_map>
#include <string>
#include "Matrix.hpp"

class NeuralNetwork {
public:
  NeuralNetwork(Matrix &X, Matrix &Y) {};
  
private:
  std::unordered_map<std::string, Matrix> weights;
  std::unordered_map<std::string, Matrix> baises;
  std::unordered_map<std::string, Matrix> Nodes;
  std::vector<int> layers {2, 3, 4, 3, 1}; // Hidden Layers Structure

  void sigmoid(Matrix &Object);
  void relu(Matrix &Object);
  void leakyrelu(Matrix &Object, double aplha = 0.01);

  void setdefault();
  void feedforward();
  void backpropogation();
  
  void analysis();
  double cost(Matrix &Node, Matrix &Y);
 
 
 
};
