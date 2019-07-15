#include "random_t.cpp"
#include <bitset>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <tuple>
#include <vector>



int
main()
{
  int i = 0;
  uint64_t state = 1;

  Random<> r;
  r.seed(10);

  while (i < 255) {
    // state = generate(state);

    //		std::cout << std::bitset<32>(state) << std::endl;
    std::cout << state << std::endl;
    std::cout << r.random() << std::endl;
    i++;
  }
}