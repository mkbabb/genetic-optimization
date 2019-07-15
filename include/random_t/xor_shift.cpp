#include <cstdint>

auto
xorshift32(uint32_t x) -> uint32_t
{
  x ^= (x << 13);
  x ^= (x >> 17);
  x ^= (x << 5);
  return x;
}

auto
xorshift64(uint64_t x) -> uint64_t
{
  x ^= (x << 13);
  x ^= (x >> 7);
  x ^= (x << 17);
  return x;
}

auto
xorshift64s(uint64_t x) -> uint64_t
{
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  return x * static_cast<uint64_t>(0x2545F4914F6CDD1D);
}

struct xorwow_state
{
  uint32_t x, y, z, w;
  uint32_t counter;
};

auto
xorwow(struct xorwow_state* state) -> uint32_t
{
  uint32_t t = state->w;

  uint32_t const s = state->x;
  state->w = state->z;
  state->z = state->y;
  state->y = s;

  t ^= t >> 2;
  t ^= t << 1;
  t ^= s ^ (s << 4);
  state->x = t;

  state->counter += 362437;
  return t + state->counter;
}
