#include <cstdint>

auto
generate(uint64_t* state) -> uint32_t
{
  uint64_t a = 25214903917UL;
  uint64_t b = 11;

  *state = (a * (*state)) + b;
  uint8_t shift = (29 - ((*state) >> 61));
  return *state >>= shift;
}

auto
lcg_cpp11(uint64_t* state) -> uint32_t
{
  uint64_t a = 48271;
  uint64_t b = 0;
  uint64_t m = 0x7FFFFFFF;
  return *state = ((a * static_cast<uint64_t>(*state)) + b) % m;
}

auto
lcg_java_util_random(uint64_t* state) -> uint32_t
{
  uint64_t a = 0x5DEECE66D;
  uint64_t b = 11;
  *state = (a * (*state)) + b;
  return *state &= a - 1;
}

auto
lcg_xor_rot(uint64_t* state) -> uint32_t
{
  uint64_t a = 6364136223846793005ULL;
  uint64_t b = 11;

  uint64_t t_state = *state;
  *state = (a * (*state)) + b;

  uint32_t xorshifted = ((t_state >> 18) ^ t_state) >> 27;
  uint32_t rot = t_state >> 59;
  return (xorshifted << (32 - rot)) | (xorshifted >> rot);
}

