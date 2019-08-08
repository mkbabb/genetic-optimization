#define FMT_HEADER_ONLY

#include "../fmt/format.h"
#include "../random_t/random_t.hpp"
#include "generator.hpp"
#include "itertools.hpp"

#include <chrono>
#include <experimental/coroutine>
#include <iostream>
#include <list>
#include <map>
#include <numeric>
#include <string>
#include <vector>

void
zip_tests()
{
  {
    std::vector<int> iv1 = {101, 102, 103, 104};
    std::vector<int> iv2 = {9, 8, 7, 6, 5, 4, 3, 2};
    std::vector<int> iv3 = {123, 1234, 12345};

    for (auto [i, j, k] : itertools::zip(iv1, iv2, iv3)) {};
  }
  {
    std::vector<std::string> sv1 = {
      "\nThis", "but", "different", "containers,"};
    std::vector<std::string> sv2 = {"is", "can", "types", "too?"};
    std::vector<std::string> sv3 = {
      "cool,", "we iterate through", "of", "...\n"};

    for (auto [i, j, k] : itertools::zip(sv1, sv2, sv3)) {};
  }
  {

    std::list<std::string> sl1 = {
      "Yes, we can!", "Some more numbers:", "More numbers!"};
    std::vector<std::string> sv1 = {
      "Different types, too!", "Ints and doubles.", ""};
    std::list<int> iv1 = {1, 2, 3};
    std::vector<double> dv1{3.141592653589793238, 1.6181, 2.71828};

    for (auto [i, j, k, l] : itertools::zip(sl1, sv1, iv1, dv1)) {}
  }
  {
    std::map<int, int> id1 = {
      {0, 10}, {1, 11}, {2, 12}, {3, 13}, {4, 14}, {5, 15}};
    std::list<std::string> sv = {"1", "mijn", "worten", "2", "helm", "dearth"};
    std::vector<double> dv = {1.2, 3.4, 5.6, 6.7, 7.8, 8.9, 9.0};

    for (auto [i, j, k, l] : itertools::zip(id1, sv, dv, itertools::range(7))) {
      auto [key, value] = i;
      //   fmt::print("{}: {}, {}, {}, {}", key, value, j, k, l);
    }
  }
  {
    std::vector<int> iv1(10, 10);
    std::vector<int> iv2(10, 10);

    auto tup = std::make_tuple(iv1, iv2);

    for (auto [i, j] : itertools::zip(tup)) {}
  }
}

void
tupletools_tests()
{
  {
    auto tup1 = std::make_tuple(1, 2, 3, 4);
    assert(tupletools::to_string(tup1) == "(1, 2, 3, 4)");
  }
  {
    auto tup1 = tupletools::make_tuple_of<20>(true);
    assert(tupletools::tuple_size<decltype(tup1)>::value == 20);
  }
}

void
itertools_tests()
{
  {
    std::vector<std::string> sv1 = {"h", "e", "l", "l", "o"};
    std::vector<int> iv1 = {1, 2, 3, 4, 5, 6, 7, 8};

    assert(itertools::join(sv1, "") == "hello");
    assert(itertools::join(sv1, ",") == "h,e,l,l,o");
    assert(itertools::join(iv1, ", ") == "1, 2, 3, 4, 5, 6, 7, 8");
  }
}

void
any_tests()
{
  {
    // Any tests with initializer list
    auto tup1 = std::make_tuple(1, 2, 3, 4, 5, 6);
    auto tup2 = std::make_tuple(33, 44, 77, 4, 5, 99);

    auto tup_bool =
      tupletools::where([](auto x, auto y) { return x == y; }, tup1, tup2);
    bool bool1 = tupletools::any_of(tup_bool);

    assert(bool1 == true);
  }
  {
    // Any tests with tuple of booleans.
    auto tup_bool1 = std::make_tuple(true, true, true, true, false);
    auto tup_bool2 = std::make_tuple(false, false, false, false, false);

    bool bool1 = tupletools::any_of(tup_bool1);
    bool bool2 = tupletools::any_of(tup_bool2);

    assert(bool1 == true);
    assert(bool2 == false);
  }
  {
    // All tests
    auto tup1 = std::make_tuple(1, 2, 3, 4, 5, 6);
    auto tup2 = std::make_tuple(1, 2, 3, 4, 5, 6);

    auto tup_bool =
      tupletools::where([](auto x, auto y) { return x == y; }, tup1, tup2);
    bool bool1 = tupletools::all_of(tup_bool);

    assert(bool1 == true);
  }
  {
    // All tests with tuple of booleans.
    auto tup_bool1 = std::make_tuple(true, true, true, true, false);
    auto tup_bool2 = std::make_tuple(true, true, true, true, true);

    bool bool1 = tupletools::all_of(tup_bool1);
    bool bool2 = tupletools::all_of(tup_bool2);

    assert(bool1 == false);
    assert(bool2 == true);
  }
  {
    // Disjunction tests
    auto tup1 = std::make_tuple(1, 2, 3, 4);
    auto tup2 = std::make_tuple(1, 2, 7, 4);

    auto ilist =
      tupletools::where([](auto x, auto y) { return x == y; }, tup1, tup2);

    bool bool1 = tupletools::disjunction_of(ilist);
    assert(bool1 == false);
  }
  {
    // Disjunction tests with tuple of booleans.
    auto tup_bool1 = std::make_tuple(true, true, true, false, false);
    auto tup_bool2 = std::make_tuple(true, true, false, true, true);

    bool bool1 = tupletools::disjunction_of(tup_bool1);
    bool bool2 = tupletools::disjunction_of(tup_bool2);

    assert(bool1 == true);
    assert(bool2 == false);
  }
}

void
enumerate_tests()
{
  {
    std::vector<int> v1(100000, 0);
    int j = 0;
    int k = 0;

    for (auto [n, i] : itertools::enumerate(v1)) {
      j++;
      k = n;
    }
    assert((j - 1) == k);
  }
  {
    std::vector<int> v1(1000000, 0);
    int j = 0;
    int k = 0;

    for (auto [n, i] : itertools::enumerate(v1)) {
      j++;
      k = n;
    }
    assert((j - 1) == k);
  }
}

template<std::size_t iterations = 1,
         class... Funcs,
         const std::size_t N = sizeof...(Funcs)>
auto
time_multiple(Funcs&&... funcs)
  -> std::map<int, std::vector<std::chrono::microseconds>>
{
  std::map<int, std::vector<std::chrono::microseconds>> times;
  std::map<int, std::vector<std::chrono::microseconds>> extremal_times;
  for (int i = 0; i < N; i++) {
    times[i] = std::vector<std::chrono::microseconds>{};
  }
  auto tup = std::make_tuple(funcs...);

  auto func = [&](auto&& n, auto&& v) {
    auto start = std::chrono::high_resolution_clock::now();
    v();
    auto stop = std::chrono::high_resolution_clock::now();
    auto time =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    times[n].push_back(time);
    return false;
  };
  for (int i = 0; i < iterations; i++) { tupletools::for_each(tup, func); }

  return times;
}

void
rvalue_zip_tests()
{
  {
    std::vector<int> v1(10, 10);

    for (auto [i, j] : itertools::zip(v1, std::vector<int>{1, 2, 3, 4})) {
      tupletools::const_downcast(i) = 99;
    }
    for (auto [n, i] : itertools::enumerate(v1)) {
      if (n < 4) { assert(i == 99); }
    }
    itertools::for_each(v1, [](auto&& n, auto&& v) {
      tupletools::const_downcast(v) = 77;
      return false;
    });
    for (auto [n, i] : itertools::enumerate(v1)) { assert(i == 77); }
  }
}

void
range_tests()
{
  {
    int stop = -999'999;
    int j = stop;
    auto _range = itertools::range(stop);
    for (auto i : _range) {
      assert(i == j);
      j++;
    }
    assert(j == 0);
  }
  {
    int stop = 1'999'999;
    int j = 0;
    auto _range = itertools::range(stop);
    for (auto i : _range) {
      assert(i == j);
      j++;
    }
    assert(j == stop);
  }

  {
    int stop = -999'999;
    int j = stop;
    auto _range = itertools::range(stop, 0);
    for (auto i : _range) {
      assert(i == j);
      j++;
    }
    assert(j == 0);
  }
}

itertools::generator<int>
inc(int n)
{
  for (int i = 0; i < n; i++) { co_yield i; };
};

itertools::generator<int>
rec(int n)
{
  co_yield n;
  if (n > 0) { co_yield rec(n - 1); }
  co_return;
};

void
generator_tests()
{
  {
    int n = 500'000;
    auto gen = rec(n);
    for (auto i : gen) { assert((n--) == i); }
  }
  {
    int n = 500'000;
    auto gen = inc(n);
    n = 0;
    for (auto i : gen) { assert((n++) == i); }
  }
}

int
main()
{
  zip_tests();
  any_tests();
  enumerate_tests();
  range_tests();
  rvalue_zip_tests();
  itertools_tests();
  tupletools_tests();
  generator_tests();

  auto mijn_func1 = []() {
    for (int i = 0; i < 1'000'000; i++) {};
  };

  auto mijn_func2 = []() {
    for (int i = 0; i < 1'000'000; i++) {};
  };

  auto times = time_multiple<10>(mijn_func1, mijn_func2);
  for (auto [key, value] : times) {
    // fmt::print("function {}:\n", key);
    for (auto time : value) {
      // fmt::print("\t{}\n", time.count());
    }
  }

  auto tup1 = std::make_tuple(37);
  const auto tup2 = std::make_tuple(1, tup1);
  auto tup3 = std::make_tuple(1, 2, tup2, 3, tup2, std::make_tuple(99));

  std::ostringstream oss;

  auto tt = tupletools::flatten(tup3);
  auto mijn = std::make_tuple(1, 2, 3, 1.4);

  tupletools::for_each(mijn, [&oss](auto&& n, auto&& v) {
    oss << v << ", ";
    v += 1;
  });

  fmt::print("{}\n", oss.str());

  fmt::print("tests complete\n");
  return 0;
}