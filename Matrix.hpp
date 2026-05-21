#include <vector>
#include <random>
#include <cmath>
#include <stdexcept>
#include <iostream>


class Matrix {

public:
  int row, column;

  Matrix ();
  
  Matrix(int row, int column): row(row), column(column) {}
  
  Matrix(const Matrix &Object);

  Matrix transpose() const;
  
  Matrix& operator=(const Matrix &Object) const
    
  Matrix operator+(const Matrix &Object) const;

  Matrix operator-(const Matrix &Object) const;

  Matrix operator-() const;
  
  Matrix operator*(const Matrix &Object) const;
  
  friend std::ostream& operator<<(std::ostream &os, const Matrix &Object) const;
  
  
  
private:
  double he_random_generator(int fan_in);

  std::vector<std::vector<double>> matrix;
};
