#include <vector>
#include <unordered_map>


using Matrix = std::vector<std::vector<double>>;
class NeuralNetwork {
public:
  NeuralNetwork() {};
  
private:
  std::unordered_map<std::string, Matrix> weights;
  std::unordered_map<std::string, Matrix> baises;
  std::unordered_map<std::string, Matrix> Nodes;
  std::vector<int> layers {2, 3, 4, 3, 1}; // Hidden Layers Structure

  Matrix generator(int row, int column);
  Matrix
  double he_random_generator(int fan_in);
  double random_generator();
 
};
