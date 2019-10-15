#ifndef MERSENNE_TWISTER_H
#define MERSENNE_TWISTER_H

#include <cstdint>
#include <vector>

#pragma once

namespace random_v {

template<typename CastType = uint32_t>
class MT19937_N
{

private:
  int W, N, M, R, NM, U, S, T, I;
  unsigned long A, D, B, C, F, LOWER_MASK;

public:
  unsigned long UPPER_MASK, divisor;
  int index, bit_size;
  CastType* state;

  explicit MT19937_N(CastType seed = 5489);

  ~MT19937_N() { delete[] state; };

  CastType operator()();

  void get_bit_size(int bit_size);

private:
  void generate();
};
}

#endif // MERSENNE_TWISTER_H
