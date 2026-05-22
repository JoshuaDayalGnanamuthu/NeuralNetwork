#include <vector>
#include <random>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <iomanip>


class Matrix {

public:
  size_t row, column;

  Matrix(): row(0), column(0) {}
  
  Matrix(size_t row, size_t column);

  Matrix(size_t row, size_t column, double value);
  
  Matrix(const Matrix &Object);

  Matrix transpose() const;
  
  Matrix& operator=(const Matrix &Object);
    
  Matrix operator+(const Matrix &Object) const;

  Matrix operator-(const Matrix &Object) const;
  
  Matrix operator*(const Matrix &Object) const;
  
  std::vector<double>& operator[](size_t index);

  const std::vector<double>& operator[](size_t index) const;

  friend std::ostream& operator<<(std::ostream &os, const Matrix &Object);
  
  
  
private:
  double he_random_generator(int fan_in);

  std::vector<std::vector<double>> matrix;
};
