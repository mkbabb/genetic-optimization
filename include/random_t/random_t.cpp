#include "random_t.hpp"

template<typename Dtype, typename CastType>
Random<Dtype, CastType>::Random()
{
  mt = nullptr;
}

template<typename Dtype, typename CastType>
Random<Dtype, CastType>::~Random()
{
  delete mt;
}

template<typename Dtype, typename CastType>
void
Random<Dtype, CastType>::init_mt()
{
  if (mt == nullptr) {
    mt = new MT19937_N<CastType>(_seed);
  };
}

template<typename Dtype, typename CastType>
void
Random<Dtype, CastType>::seed(CastType seed)
{
  _seed = seed;
  if (mt != nullptr) {
    mt = new MT19937_N<CastType>(seed);
  } else {
    delete mt;
    mt = new MT19937_N<CastType>(seed);
  }
}

template<typename Dtype, typename CastType>
CastType
Random<Dtype, CastType>::random()
{
  init_mt();

  return mt->operator()();
}

template<typename Dtype, typename CastType>
CastType
Random<Dtype, CastType>::random(DriverType driver)
{
  init_mt();

  return mt->operator()();
}

template<typename Dtype, typename CastType>
Dtype
Random<Dtype, CastType>::db_modulo_twice(CastType bound)
{
  CastType t = (-bound) % bound;
  CastType r;

  while (true) {
    r = random();
    if (r >= t) {
      return r % bound;
    };
  }
}

template<typename Dtype, typename CastType>
Dtype
Random<Dtype, CastType>::unit()
{
  init_mt();

  return random() / ((Dtype)mt->UPPER_MASK * 2);
}

template<typename Dtype, typename CastType>
std::vector<Dtype>
Random<Dtype, CastType>::randrange(Dtype a, Dtype b, size_t N)
{
  init_mt();

  Dtype val;

  CastType range = b - a;

  std::vector<Dtype> nums(N, 0);
  for (int i = 0; i < N; i++) {
    nums[i] = db_modulo_twice(range) + a;
  }
  return nums;
}

template<typename Dtype, typename CastType>
std::vector<Dtype>
Random<Dtype, CastType>::randrange(size_t N, DriverType driver)
{
  init_mt();

  Dtype val;
  std::vector<Dtype> nums(N, 0);
  for (int i = 0; i < N; i++) {
    nums[i] = driver();
  }
  return nums;
}

template<typename Dtype, typename CastType>
Dtype
Random<Dtype, CastType>::uniform(const Dtype a, const Dtype b)
{
  init_mt();

  return random() / ((mt->divisor) / (b - a)) + a;
}
