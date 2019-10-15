#include <cstdint>

namespace random_v {
uint32_t
xorshift32(uint32_t* state)
{
  *state ^= (*state << 13);
  *state ^= (*state >> 17);
  *state ^= (*state << 5);
  return *state;
}

uint64_t
xorshift64(uint64_t* state)
{
  *state ^= (*state << 13);
  *state ^= (*state >> 7);
  *state ^= (*state << 17);
  return *state;
}

uint64_t
xorshift64s(uint64_t* state)
{
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state) * static_cast<uint64_t>(0x2545F4914F6CDD1D);
}

struct xorwow_state
{
  uint32_t state, y, z, w;
  uint32_t counter;
};

uint32_t
xorwow(struct xorwow_state* state)
{
  uint32_t t = state->w;

  uint32_t const s = state->state;
  state->w = state->z;
  state->z = state->y;
  state->y = s;

  t ^= t >> 2;
  t ^= t << 1;
  t ^= s ^ (s << 4);
  state->state = t;

  state->counter += 362437;
  return t + state->counter;
}
};
