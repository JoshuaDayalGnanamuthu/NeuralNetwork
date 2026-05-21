#include "Matrix.h"

/**
 * Constructor to generate a Matrix of size (row x column)
 * of random doubles [-1, 1] with uniform distribution
 * @param row The number of rows of the Matrix
 * @param column The number of columns of the Matrix
 */
Matrix::Matrix(int row, int column): row(row), column(column) {
  for ( size_t i {}; i < row; ++i ) {
    std::vector<double> r;
    for ( size_t j {}; j < column; ++j ) {
      double randomNumber = he_random_generator(column);
      r.push_back(randomNumber);
    }
    this->matrix.push_back(r);
  }
}

/**
 * Constructor to generate a Matrix of size (row x column)
 * populated with a single value
 * @param row The number of rows of the Matrix
 * @param column The number of columns of the Matrix
 * @param value The value to assign to all entires
 */
Matrix::Matrix(int row, int column, double value): row(row), column(column) {
  for ( size_t i {}; i < row; ++i ) {
    std::vector<double> r(column, value);
    this->matrix.push_back(r);
  }
}

/**
 * Constructor to create a deepcopy of a Matrix
 * @param Object A reference to the Matrix object to be copied 
 */
Matrix::Matrix(const Matrix&Object) {
  this->row = Object.row;
  this->column = Object.column;

  for ( int i {}; i < this->row; ++i ) {
    std::vector<double> new_row;
    for ( int j {}; j < this->column; ++j ) {
      new_row.push_back(Object.matrix[i][j]);
    }
    this->matrix.push_back(new_row);
  }
}

/**
 * Uniform Random Float Generator (approx ~ [-1, 1])
 * @param fan_in The seed
 * @return a random double 
 */				     
double Matrix::he_random_generator(int fan_in) {
  double stdev = std::sqrt(2.0 / fan_in);
  std::normal_distribution<double> dist(0.0, stdev);
  return dist(re);
}

/**
 * Computes and returns the transpose of a given Matrix
 * @return The transposed Matrix
 */
Matrix Matrix::transpose() const {
  Matrix Object{};
  Object.row = this->column; Object.column = this->row;
  
  for ( size_t i {}; i < this->column; ++i ) {
    std::vector<double> new_row;
    for ( size_t j {}; j < this->row; ++j ) {
      new_row.push_back(this->matrix[j][i]);
    }
    Object.matrix.push_back(new_row);
  }
  return Matrix;
}

/**
 * Overloads the assigment operator to copy the contens of
 * one Matrix to another
 * @param Object A const reference to the Matrix to be copied over
 * @return A reference to the Matrix that got copied into
 */
Matrix& Matrix::operator=(const Matrix &Object) const {
  this->row = Object.row;
  this->column = Object.column;
  this->matrix.clear();

  for ( size_t i {};  i < this->row; ++i ) {
    std::vector<double> new_row;
    for ( size_t j {}; j < this->column; ++j ) {
      new_row.push_back(Object.matrix[i][j]);
    }
    this->matrix.push_back(new_row);
  }
  return *this;
}

/**
 * Overloads the addition operator to compute the sum of
 * two Matrices
 * @param Object A const reference to the right hand operand
 * @return A Matrix Object representing the resultant sum 
 */
Matrix Matrix::operator+(const Matrix &Object) const {
  try {
    if (this->row != Object.row || this->column != Object.column) {
      throw std::invalid_arguement("Incompatible Matrix Sizes for Addition.\n");
    }
    Matrix Sum{};
    Sum.row = this->row; Sum.column = this->column;

    for ( size_t i {}; i < this->row; ++i ) {
      std::vector<double> new_row;
      for ( size_t j {}; j < this->column; ++j ) {
	new_row.push_back(this->matrix[i][j] + Object.matrix[i][j]);
      }
      Sum.matrix.push_back(new_row);
    }

    return Sum;
  }

  catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << std::endl;
  }
}

/**
 * Overloads the subtraction operator to compute the difference of
 * two Matrices
 * @param Object A const reference to the right hand operand
 * @return A Matrix Object representing the resultant difference
 */
Matrix Matrix::operator-(const Matrix &Object) const {                              
  try {                                                                             
    if (this->row != Object.row || this->column != Object.column) {                 
      throw std::invalid_arguement("Incompatible Matrix Sizes for Addition.\n");    
    }                                                                               
    Matrix Sum{};
    Sum.row = this->row; Sum.column = this->column;
    
    for ( size_t i {}; i < this->row; ++i ) {                                       
      std::vector<double> new_row;                                                  
      for ( size_t j {}; j < this->column; ++j ) {                                  
        new_row.push_back(this->matrix[i][j] - Object.matrix[i][j]);                
      }                                                                             
      Sum.matrix.push_back(new_row);                                                
    }                                                                                   return Sum;                                                                     
  }                                                                                 
                                                                                    
  catch (const std::exception &e) {                                                 
    std::cerr << "Exception: " << e.what() << std::endl;                            
  }                                                                                 
}  

/**
 * Overloads the multiplication operator to compute the Matrix multiplication
 * of two Matrices
 * @param Object A const reference to the right hand operand
 * @return A Matrix Object representing the resultant product
 */
Matrix Matrix::operator*(const Matrix &Object) const {
  Matrix Object_T = Object.transpose();
  int row = this->row, column = Object_T.row;
  Matrix Product{};
  Product.row = row; Product.column = column;

  for ( size_t i {}; i < row; ++i ) {
    std::vector<double> new_row;
    std::vector<double> vec1 = this->matrix[i];
    for ( size_t j {}; j < column; ++j ) {
      std::vector<double> vec2 = Object_T.matrix[j];
      double sum;

      if (vec1.size() != vec2.size()) {
	throw std::invalid_argument("Incompatible Matrix Sizes for Multiplication.\n");
      }
      for ( size_t k {}; k < vec1.sie(); ++k ) {
	sum += vec1[k] + vec2[k];
      }
      new_row.push_back(sum);
    }
    Product.matrix.push_back(new_row);
  }
  return Product;
}

/**
 * overloads the [] operator to return the row corresponding to the index
 * @param index The index of the corresponding row (zero indexed)
 * @return A vector of doubles corresponding to the indexed row
 */
std::vector<double>& Matrix::operator[](int index) {
  if (index < 0 || index >= this->row) {
    throw std::invalid_arguement("Index Out of Bounds.\n");
  }
  return this->matrix[index];
}

/**
 * overloads the << operator to print the Matrix object to the 
 * output stream
 * @param os A reference to the output stream buffer
 * @param Object A const reference to the Matrix Objet to be printed
 * @return A reference to the passed output stream buffer
 */
std::ostream& operator<<(std::ostream &os, const Matrix &Object) const {
  for ( size_t i {}; i < Object.row; ++i ) {
    os << "[";
    for ( size_t j {}; j < Object.column; ++j ) {
      if (j != Object.column - 1) {
	os << std::fixed << std::setprecision(5) << Object[i][j] << ",";
      }
      else {
	os << std::fixed << std::setprecision(10) << Object[i][j] << " ";
      }
    }
    os << "]" << std::endl;
  }
  return os;
}
  
  
