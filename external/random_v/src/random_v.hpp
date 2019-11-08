#ifndef RAND_H
#define RAND_H

#include "lcg.cpp"
#include "mersenne_twister.cpp"
#include "xor_shift.cpp"

#include <cmath>
#include <cstdint>

#pragma once

namespace random_v {

template<typename StateType = uint64_t,
         typename OutputType = uint32_t,
         typename DriverType = void>
class Random
{
    StateType _state;
    DriverType& _rng;

  public:
    Random(StateType state)
      : _state(state)
    {
        _rng = lcg_xor_rot;
    };
    Random(StateType* state)
      : Random(*state){};

    Random(DriverType& rng)
      : _state(static_cast<uint64_t>(1))
      , _rng(rng){};

    Random(StateType state, DriverType& rng)
      : _state(state)
      , _rng(rng){};

    Random(StateType* state, DriverType& rng)
      : Random(*state, rng){};

    Random(StateType state, DriverType& rng, OutputType out_state)
      : _state(state)
      , _rng(rng){};

    StateType state() { return _state; };
    StateType state() const { return _state; };
    void state(StateType state) { _state = state; };

    OutputType generate()
    {
        OutputType t = _rng(&_state);
        return t;
    };

    OutputType operator()() { return this->generate(); };

    OutputType rand2m(uint8_t m)
    {
        return this->generate() >> (sizeof(OutputType) - (m + 1));
    };

    OutputType bounded_rand(OutputType range)
    {
        OutputType t = (-range) % range;
        while (true) {
            OutputType r = this->generate();
            if (r >= t)
                return r % range;
        }
    };

    OutputType randrange(OutputType a, OutputType b)
    {
        OutputType range = b - a;
        return bounded_rand(range) + a;
    };

    double unit(OutputType interval = 0xFFFFFFFF)
    {
        double i = static_cast<double>(interval);
        return this->generate() / i;
    };
};
}
#endif // RAND_H