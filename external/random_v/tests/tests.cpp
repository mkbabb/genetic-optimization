#define FMT_HEADER_ONLY

#include "../../cxxopts.hpp"
#include "../../fmt/format.h"
#include "../../itertools/src/itertools.hpp"
#include "../src/random_v.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <random>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <vector>

using namespace random_v;

template<class Func, class Output = typename std::result_of<Func(void)>::type>
double
rand_loop(Func rng, int N, bool print_loop = false)
{
  std::map<Output, double> uniform_test_dict;
  double avg_total = 0;
  double stdev = 0;

  for (int i = 0; i < N; i++) {
    Output r = rng();
    uniform_test_dict[r] += 1;
  }
  for (auto& [k, v] : uniform_test_dict) {
    double avg = static_cast<double>(v) / N;
    if (print_loop) { fmt::print("n: {0}, count: {1}, avg: {2}\n", k, v, avg); }
    avg_total += avg;
    uniform_test_dict[k] = avg;
  }
  avg_total /= uniform_test_dict.size();
  for (auto& [k, v] : uniform_test_dict) {
    double t = std::pow((v - avg_total), 2);
    stdev += t;
  }
  stdev = std::sqrt(stdev / (uniform_test_dict.size() - 1));
  fmt::print("\n\tstdev: {0:.10f}\n", stdev);

  return stdev;
}

void
rand_tests(int a, int b, int N)
{
  {
    MT19937_N _mt(5489);
    uint64_t state1 = 0xDEADBEEF;

    auto mt = [&](auto s) { return _mt(); };
    Random r1(state1, mt);

    fmt::print("\n--- MT19937; randrange ---\n");
    rand_loop([&]() { return r1.randrange(a, b); }, N);
    fmt::print("\n--- MT19937; 2m ---\n");
    rand_loop([&]() { return r1.rand2m(8); }, N);
  }
  {
    uint64_t state1 = 0xDEADBEEF;
    Random r1(state1, lcg_xor_rot);

    fmt::print("\n--- lcg_xor_rot; randrange ---\n");
    rand_loop([&]() { return r1.randrange(a, b); }, N);
    fmt::print("\n--- lcg_xor_rot; 2m ---\n");
    rand_loop([&]() { return r1.rand2m(8); }, N);
  }
  {

    uint64_t state1 = 0xDEADBEEF;
    Random r1(state1, minstd_rand);

    fmt::print("\n--- minstd_rand; randrange ---\n");
    rand_loop([&]() { return r1.randrange(a, b); }, N);
    fmt::print("\n--- minstd_rand; 2m ---\n");
    rand_loop([&]() { return r1.rand2m(8); }, N);
  }
  {
    uint64_t state1 = 0xDEADBEEF;
    Random r1(state1, lcg_0);

    fmt::print("\n--- lcg_0; randrange ---\n");
    rand_loop([&]() { return r1.randrange(a, b); }, N);
    fmt::print("\n--- lcg_0; 2m ---\n");
    rand_loop([&]() { return r1.rand2m(8); }, N);
  }
  {
    uint64_t state1 = 0xDEADBEEF;
    Random r1(state1, xorshift64s);

    fmt::print("\n--- xorshift64s; randrange ---\n");
    rand_loop([&]() { return r1.randrange(a, b); }, N);
    fmt::print("\n--- xorshift64s; 2m ---\n");
    rand_loop([&]() { return r1.rand2m(8); }, N);
  }
}

int
main(int argc, char** argv)
{
  cxxopts::Options options("rand_tests", "to test rand");
  options.allow_unrecognised_options().add_options()(
    "a", "Interval start", cxxopts::value<int>())(
    "b", "Interval end", cxxopts::value<int>())(
    "N", "Iteration count", cxxopts::value<int>());

  auto result = options.parse(argc, argv);

  int a = result["a"].as<int>();
  int b = result["b"].as<int>();
  int N = result["N"].as<int>();

  rand_tests(a, b, N);
}