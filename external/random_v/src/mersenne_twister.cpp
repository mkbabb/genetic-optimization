#include "mersenne_twister.hpp"

using namespace random_v;

template<typename CastType>
void
MT19937_N<CastType>::get_bit_size(int bit_size)
{
  if (bit_size == 32) {
    W = 32;
    N = 624;
    M = 397;
    R = 31;

    NM = N - M;

    UPPER_MASK = 0x80000000;
    LOWER_MASK = 0x7FFFFFFF;

    A = 0x9908B0DF;

    U = 11;
    D = 0xFFFFFFFF;

    S = 7;
    B = 0x9D2C5680;

    T = 15;
    C = 0xEFC60000;

    I = 18;

    F = 0x6C078965;
    divisor = 0x100000000;

  } else if (bit_size == 64) {

    W = 64;
    N = 312;
    M = 156;
    R = 31;

    NM = N - M;

    UPPER_MASK = 0x8000000000000000;
    LOWER_MASK = 0x7FFFFFFFFFFFFFFF;

    A = 0xB5026F5AA96619E9;

    U = 29;
    D = 0x5555555555555555;

    S = 17;
    B = 0x71D67FFFEDA60000;

    T = 37;
    C = 0xFFF7EEE000000000;

    I = 43;

    F = 0x5851F42D4C957F2D;
    divisor = LONG_MAX;
  };
}

template<typename CastType>
MT19937_N<CastType>::MT19937_N(CastType seed)
  : index(0)
{

  bit_size = typeid(CastType) == typeid(unsigned long) ? 64 : 32;
  get_bit_size(bit_size);

  state = new CastType[N];
  state[0] = seed;

  for (int i = 1; i < N; i++) {
    state[i] = (CastType)(F * (state[i - 1] ^ (state[i - 1] >> (W - 2)))) + i;
  }
  generate();
};

template<typename CastType>
CastType
MT19937_N<CastType>::operator()()
{
  if (index >= M)
    generate();
  unsigned long y = state[index];
  index++;

  y ^= (y >> U) & D;
  y ^= (y << S) & B;
  y ^= (y << T) & C;
  y ^= y >> I;

  return (CastType)y;
}

template<typename CastType>
void
MT19937_N<CastType>::generate()
{
  int i;
  CastType bits;

  for (i = 0; i < NM; i++) {
    bits = (state[i] & UPPER_MASK) | (state[i + 1] & LOWER_MASK);
    state[i] = state[i + M] ^ (bits >> 1) ^ ((bits & 1) * A);
  }
  for (; i < N - 1; i++) {
    bits = (state[i] & UPPER_MASK) | (state[i + 1] & LOWER_MASK);
    state[i] = state[i - NM] ^ (bits >> 1) ^ ((bits & 1) * A);
  }

  bits = (state[i] & UPPER_MASK) | (state[0] & LOWER_MASK);
  state[i] = state[M - 1] ^ (bits >> 1) ^ ((bits & 1) * A);
  index = 0;
}