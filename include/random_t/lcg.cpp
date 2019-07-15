#include <cstdint>

uint32_t
generate(uint64_t state)
{
  uint64_t a = 25214903917UL;
  uint64_t b = 11;

  uint64_t t = (a * state) + b;
  t >>= (29 - (state >> 61));
  return static_cast<uint32_t>(t);
}

auto
lcg_cpp11(uint64_t state) -> uint32_t
{
  uint64_t a = 48271;
  uint64_t b = 0;
  uint64_t m = 0x7FFFFFFF;

  uint64_t t = ((a * state) + b) % m;

  return static_cast<uint32_t>(t);
}

auto
lcg_java_until_random(uint64_t state) -> uint32_t
{
  uint64_t a = 0x5DEECE66D;
  uint64_t b = 11;

  uint64_t t = (a * state) + b;
  t &= (a - 1);

  return static_cast<uint32_t>(t);
}