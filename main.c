#include <stdio.h>
#include <string.h>


int main(int argc, char const *argv[])
{
    int a;
    printf("Enter the number whose multiplication table you whish: ");
    scanf("%d", &a);

    if (a < 0) printf("false");


    for (size_t i = 1; i < 11; i++)
    {
        printf("%d x %zu = %zu\n", a, i, a * i);
    }
    
    

    return 0;
}
