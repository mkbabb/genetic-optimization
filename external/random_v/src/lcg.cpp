#include <cstdint>

namespace random_v {
uint32_t
lcg_0(uint64_t* state)
{
  uint64_t a = 25214903917UL;
  uint64_t b = 11;

  *state = (a * (*state)) + b;
  uint8_t shift = (29 - ((*state) >> 61));
  *state >>= shift;

  uint32_t out = *state;
  return out;
}

uint32_t
minstd_rand(uint64_t* state)
{
  uint64_t a = 48271;
  uint64_t b = 0;
  uint64_t m = 0x7FFFFFFF;
  *state = ((a * (*state)) + b) % m;

  uint32_t out = *state;
  return out;
}

uint32_t
java_util_random(uint64_t* state)
{
  uint64_t a = 0x5DEECE66D;
  uint64_t b = 11;
  *state = (a * (*state)) + b;
  *state &= a - 1;

  uint32_t out = *state;
  return out;
}

uint32_t
lcg_xor_rot(uint64_t* state)
{
  uint64_t a = 1144453172214656471ULL;
  uint64_t b = 17;

  uint64_t t_state = *state;
  *state = (a * (*state)) + b;

  uint64_t xorshifted = t_state;
  xorshifted ^= (xorshifted >> 18U);
  xorshifted ^= (xorshifted >> 27U);
  xorshifted *= 0x2545F4914F6CDD1DULL;
  xorshifted >>= 32;

  uint32_t rot = t_state >> 59U;
  uint32_t out = (xorshifted << (32U - rot)) | (xorshifted >> rot);

  return out;
}

// uint32_t
// lcg_xor_rot(uint64_t* state)
// {
//   uint64_t a = 6364136223846793005ULL;
//   uint64_t b = 11;

//   uint64_t t_state = *state;
//   *state = (a * (*state)) + b;

//   uint32_t xorshifted = ((t_state >> 18) ^ t_state) >> 27;
//   uint32_t rot = t_state >> 59;
//   return (xorshifted << (32 - rot)) | (xorshifted >> rot);
// }
}
