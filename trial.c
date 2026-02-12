#include <stdio.h>
#include <string.h>
#include <ctype.h>

int main(void) {
  char str[8] = "jays!";
  int size = (int) strlen(str);
  int i = size - 1;
  while (i > 0) {
    printf("%s\n", str);
    str[i++] = toupper(str[i]);
    i /= 2;
  }
  str[3] = '\0';
  printf("%s\n", str);
  return 0;
}