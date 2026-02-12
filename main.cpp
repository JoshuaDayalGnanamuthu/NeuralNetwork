#include <iostream>

int main(int argc, char const *argv[])
{   
    int marks[5] = {22, 33, 56, 45, 56};
    int smallest = INT_MAX;

    for (int &i: marks){
        i *= 10;
        std::cout << i << std::endl;
    }



    return 0;
}
