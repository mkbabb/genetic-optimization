#ifndef RANDOM_H
#define RANDOM_H

#include "mersenne_twister.cpp"
#include <cstdint>
#include <functional>
#include <vector>

template<typename Dtype = int, typename CastType = uint32_t>
class Random
{
private:
  CastType _seed = 5489;
  MT19937_N<CastType>* mt;

  void init_mt();

public:
  typedef std::function<Dtype()> DriverType;

  Random();

  ~Random();

  void seed(CastType seed);

  CastType random();
  CastType random(DriverType d);

  Dtype unit();
  Dtype db_modulo_twice(CastType range);

  std::vector<Dtype> randrange(Dtype a, Dtype b, size_t N);
  std::vector<Dtype> randrange(size_t N, DriverType d);

  Dtype uniform(Dtype a = 0, Dtype b = 1);
};

#endif // RANDOM_H
