#include<iostream>
#include<vector>
#include<cmath>
#include<random>
#include<iomanip>
#include<string>
#include<unordered_map>
#include <thread>
#include <chrono>
#include <fstream>
#include <sstream>
#include <stdexcept>


using Matrix = std::vector<std::vector<double>>;
std::unordered_map<std::string, Matrix> Weights;
std::unordered_map<std::string, Matrix> Baises;
std::unordered_map<std::string, Matrix> Nodes;
std::vector<int> layers {2, 3, 4, 3, 1};

Matrix X {{150, 254, 312, 120, 154, 212, 216, 145, 184, 130}, //weight
          {70, 73, 68, 60, 61, 65, 67, 67, 64, 69}}; //height

Matrix Y {{0, 1, 1, 0, 0, 1, 1, 0, 1, 0}};



void sigmoid(Matrix &MyVector){
	for (auto &row: MyVector){
		for (double &value: row){
			value = 1.0 /(1.0 + exp(-value));
		}
	}
}


void relu(Matrix &MyVector) {
	for (auto &row: MyVector){
		for (double &value: row){
			value = (value > 0.0) ? value : 0.0;
		}
	}
}


void leaky_relu(Matrix &MyVector, double alpha = 0.01) {
    for (auto &row: MyVector) {
        for (double &value: row) {
            value = (value > 0.0) ? value : alpha * value;
        }
    }
}


void print_vector(const Matrix &MyVector){
	for (auto &row: MyVector){
		std::cout << "[ ";
		for (auto &value: row){
			if (value != row[row.size() -1]) std::cout << std::fixed << std::setprecision(10) << value << ", ";
			else std::cout << std::fixed << std::setprecision(15) << value << " ";
		}
		std::cout << "]" << std::endl;
	}
}


Matrix transpose(const Matrix &MyVector){
	int row = MyVector.size();
	int column = MyVector[0].size();
	Matrix NewVector;
	for (size_t i {0}; i < column; i++){
		std::vector<double> new_row;	
		for (size_t j {0}; j < row; j++){
			new_row.push_back(MyVector[j][i]);
		}
		NewVector.push_back(new_row);
	}	
	return NewVector;	
}


std::default_random_engine re(std::random_device{}());
std::uniform_real_distribution<double> unif(-1.0, 1.0);
double random_generator(){
	return unif(re);
}


double he_random_generator(int fan_in) {
    double stddev = std::sqrt(2.0 / fan_in);
    std::normal_distribution<double> dist(0.0, stddev);
    return dist(re);
}

Matrix generator(int row, int column) {
    Matrix MyVector;
    for (size_t i = 0; i < row; i++) {
        std::vector<double> r;
        for (size_t j = 0; j < column; j++) {
            double randomNumber = he_random_generator(column);  // column = fan_in
            r.push_back(randomNumber);
        }
        MyVector.push_back(r);
    }
    return MyVector;
}


void setdefault(){
	for (size_t i {1}; i < layers.size(); i++){
		std::string Weight = "W";
		std::string Bias = "b";
		Bias += std::to_string(i);
		Weight += std::to_string(i);
		Weights[Weight] = generator(layers[i], layers[i-1]);
		Baises[Bias] = generator(layers[i], 1);
	}
}


Matrix multiply(const Matrix &MyVector1, const Matrix &MyVector2) {
    Matrix MyVector2_T = transpose(MyVector2);
    int row = MyVector1.size();
    int column = MyVector2_T.size();
    Matrix NewVector;

    for (size_t i = 0; i < row; i++){
        std::vector<double> new_row;
        std::vector<double> vec1 = MyVector1[i];
        for (size_t j = 0; j < column; j++){
            std::vector<double> vec2 = MyVector2_T[j];
            double sum = 0;
            if (vec1.size() != vec2.size()) {
                throw std::invalid_argument("Incompatible vector sizes for dot product.");
            }
            for (size_t k = 0; k < vec1.size(); k++) {
                sum += vec1[k] * vec2[k];
            }
            new_row.push_back(sum);
        }
        NewVector.push_back(new_row);
    }
    return NewVector;
}


Matrix add(const Matrix &MyVector1, const Matrix &MyVector2){
	if (MyVector1.size() != MyVector2.size()) throw std::invalid_argument("Incompatible vector sizes for addition.");
	int row = MyVector1.size();
	int column = MyVector1[0].size();
	Matrix NewVector;

	for (size_t i {0}; i < row; i++){
		if (MyVector1[i].size() != MyVector2[i].size()) throw std::invalid_argument("Incompatible vector sizes for addition.");
		std::vector<double> new_row;
		for (size_t j {0}; j < column; j++){ 
			new_row.push_back(MyVector1[i][j] + MyVector2[i][j]);
		}
		NewVector.push_back(new_row);
	}
	return NewVector;

}


Matrix subtract(const Matrix &MyVector1, const Matrix &MyVector2) {
    Matrix Myvector = MyVector1;
    for (size_t i = 0; i < MyVector1.size(); ++i)
        for (size_t j = 0; j < MyVector1[0].size(); ++j)
            Myvector[i][j] -= MyVector2[i][j];
    return Myvector;
}


Matrix broadcast(const Matrix &MyVector, size_t size){
	int row = MyVector.size();
	Matrix NewVector;
	for (size_t i {0}; i < row; i++){
		std::vector<double> new_row(size, MyVector[i][0]);
		NewVector.push_back(new_row);
	}
	return NewVector;
}


Matrix normalize_standard(const Matrix &MyVector) {
    Matrix NewVector;
    for (const auto &row : MyVector) {
        std::vector<double> new_row;
        double mean = 0.0, stddev = 0.0;
        size_t n = row.size();

        for (double val : row) mean += val;
        mean /= n;

        for (double val : row) stddev += (val - mean) * (val - mean);
        stddev = std::sqrt(stddev / n);
        if (stddev < 1e-8) stddev = 1.0;

        for (double val : row) new_row.push_back((val - mean) / stddev);
        NewVector.push_back(new_row);
    }
    return NewVector;
}


void feed_forward(){
	for (size_t i {0}; i < layers.size(); i++){
		std::string Node = "A";
		if (i == 0){
			Node += std::to_string(i);
			Nodes[Node] = normalize_standard(X);
		}
		else {
		    Node += std::to_string(i - 1);
			std::string Weight = "W";
			std::string Bias = "b";
			Bias += std::to_string(i);
			Weight += std::to_string(i);
			Matrix Z = multiply(Weights[Weight], Nodes[Node]);
			Matrix B = broadcast(Baises[Bias], Z[0].size());
			Z = add(Z, B);
			if (i != layers.size()-1) {
				leaky_relu(Z);
				Z = normalize_standard(Z);
			}
			else sigmoid(Z);
			Node = "A";
			Node += std::to_string(i);
			Nodes[Node] = Z;
		}
	}
}


void analysis() {
	std::cout << "===== Weights =====" << std::endl;
    for (auto &pair : Weights) {
        std::cout << pair.first << " (" << pair.second.size() << "x" << pair.second[0].size() << "):\n";
        print_vector(pair.second);
        std::cout << std::endl;
    }

    std::cout << "===== Biases =====" << std::endl;
    for (auto &pair : Baises) {
        std::cout << pair.first << " (" << pair.second.size() << "x" << pair.second[0].size() << "):\n";
        print_vector(pair.second);
        std::cout << std::endl;
    }

    std::cout << "===== Nodes =====" << std::endl;
    for (auto &pair : Nodes) {
        std::cout << pair.first << " (" << pair.second.size() << "x" << pair.second[0].size() << "):\n";
        print_vector(pair.second);
        std::cout << std::endl;
    }
}


double cost_function(Matrix &Node, Matrix &Y){
	const double eps = 1e-15; // small constant to avoid log(0)
	double sum {0};
	double elements = Y[0].size();
	for (size_t i {0}; i < elements; i++){
		sum -= (Y[0][i] * std::log(Node[0][i] + eps) + (1 - Y[0][i]) * std::log(1 - Node[0][i] + eps));
	}
	return sum/elements;
}


void back_propagation(const Matrix &Y, double learning_rate = 0.01) {
    std::unordered_map<std::string, Matrix> dW;
	std::unordered_map<std::string, Matrix> dB;
	std::unordered_map<std::string, Matrix> dA;
	std::unordered_map<std::string, Matrix> dZ;

    size_t L = layers.size() - 1; 
    size_t m = Y[0].size();

    std::string AL = "A" + std::to_string(L);
    dA[AL] = subtract(Nodes[AL], Y);

    for (int l = L; l >= 1; l--) {
        std::string A_curr = "A" + std::to_string(l);
        std::string A_prev = "A" + std::to_string(l - 1);
        std::string W_curr = "W" + std::to_string(l);
        std::string b_curr = "b" + std::to_string(l);

        dZ[A_curr] = dA[A_curr];
        for (size_t i = 0; i < Nodes[A_curr].size(); i++) {
            for (size_t j = 0; j < Nodes[A_curr][0].size(); j++) {
                double a = Nodes[A_curr][i][j];
                if (l == L) { 
                    dZ[A_curr][i][j] *= a * (1 - a);
                } else {      
                    dZ[A_curr][i][j] *= (a > 0 ? 1.0 : 0.01);
                }
            }
        }

        Matrix dW_mat = multiply(dZ[A_curr], transpose(Nodes[A_prev]));
        for (auto &row : dW_mat)
            for (auto &val : row)
                val /= m;
        dW[W_curr] = dW_mat;

        Matrix dB_mat(dZ[A_curr].size(), std::vector<double>(1, 0.0));
        for (size_t i = 0; i < dZ[A_curr].size(); i++) {
            double sum = 0.0;
            for (size_t j = 0; j < dZ[A_curr][0].size(); j++)
                sum += dZ[A_curr][i][j];
            dB_mat[i][0] = sum / m;
        }
        dB[b_curr] = dB_mat;

        if (l > 1) {
            dA[A_prev] = multiply(transpose(Weights[W_curr]), dZ[A_curr]);
        }
    }

    for (size_t l = 1; l < layers.size(); l++) {
        std::string W_curr = "W" + std::to_string(l);
        std::string b_curr = "b" + std::to_string(l);

        for (size_t i = 0; i < Weights[W_curr].size(); i++)
            for (size_t j = 0; j < Weights[W_curr][0].size(); j++)
                Weights[W_curr][i][j] -= learning_rate * dW[W_curr][i][j];

        for (size_t i = 0; i < Baises[b_curr].size(); i++)
            Baises[b_curr][i][0] -= learning_rate * dB[b_curr][i][0];
    }
}

int main(){
	setdefault();	
	feed_forward();
	std::cout << "Neural Network in C++" << std::endl;
	
	double cost = cost_function(Nodes["A" + std::to_string(layers.size() - 1)], Y); // Final layer node
	std::cout << "Cost: " << std::fixed << std::setprecision(15) << cost << std::endl;


	std::this_thread::sleep_for(std::chrono::seconds(0));
	size_t epochs = 10000;
	
	for (size_t i {0}; i < epochs; i++){
		feed_forward();
		back_propagation(Y, 0.01);
		if (i % 10 == 0){
        cost = cost_function(Nodes["A" + std::to_string(layers.size() - 1)], Y);
			std::cout << "==============================" << std::endl;
			std::cout << " Epoch: " << i << std::endl;
			std::cout << " Cost:  " << std::fixed << std::setprecision(15) << cost << std::endl;
			std::cout << "==============================" << std::endl;
    	}
	}

	feed_forward();
    Matrix predictions = Nodes["A" + std::to_string(layers.size() - 1)];

    std::cout << "\n=== Model Predictions vs Known Y ===\n";
    std::cout << "Predicted\tKnown\n";
    for (size_t i = 0; i < predictions[0].size(); ++i) {
        std::cout << std::fixed << std::setprecision(4)
                  << predictions[0][i] << "\t\t" << Y[0][i] << std::endl;
    }
	return 0;
}
