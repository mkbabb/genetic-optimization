#define FMT_HEADER_ONLY

#include "../fmt/format.h"
#include "../random_t/random_t.hpp"
#include "generator.hpp"
#include "itertools.hpp"

#include <experimental/coroutine>
#include <iostream>
#include <list>
#include <map>
#include <numeric>
#include <string>
#include <vector>

void
tupletools_tests()
{
  auto tup1 = std::make_tuple(1, 2, 3, 4, 5, 6);
  auto tup2 = std::make_tuple(33, 44, 77, 4, 5, 99);

  auto tup_bool =
    tupletools::where([](auto x, auto y) { return x == y; }, tup1, tup2);
  bool bool1 = tupletools::any_of(tup_bool);

  //   std::cout << std::boolalpha;
  //   std::cout << bool1 << std::endl;
}

void
tupletools_zip_tests()
{
  std::vector<int> iv1 = {1, 2};
  std::vector<int> iv2 = {99, 88, 77, 66};

  for (auto [i, j] : itertools::zip(iv1, iv2)) {
    // std::cout << i << " " << j << std::endl;
  };

  std::vector<std::string> sv1 = {"\nThis", "but", "different", "containers,"};
  std::vector<std::string> sv2 = {"is", "can", "types", "too?"};
  std::vector<std::string> sv3 = {"cool,", "we iterate through", "of", "...\n"};

  for (auto [i, j, k] : itertools::zip(sv1, sv2, sv3)) {
    // std::cout << i << " " << j << " " << k << std::endl;
  };

  std::list<std::string> sl1 = {
    "Yes, we can!", "Some more numbers:", "More numbers!"};
  std::vector<std::string> sv4 = {
    "Different types, too!", "Ints and doubles.", ""};
  std::list<int> iv3 = {1, 2, 3};
  std::vector<double> dv5{3.141592653589793238, 1.6181, 2.71828};

  for (auto [i, j, k, l] : itertools::zip(sl1, sv4, iv3, dv5)) {
    // std::cout << i << " " << j << " " << k << " " << l << std::endl;
  }

  std::vector<int> iv4 = {101, 102, 103, 104};
  std::vector<int> iv5 = {9, 8, 7, 6, 5, 4, 3, 2};
  std::vector<int> iv6 = {123, 1234, 12345};

  // std::cout << "\nAutomatically adjusts and shrinks the iterator to the
  //   "
  //                "smallest of the zipped iterator sizes!\n"
  //             << std::endl;

  for (auto [i, j, k] : itertools::zip(iv4, iv5, iv6)) {
    // std::cout << i << " " << j << " " << k << std::endl;
  };

  auto tup1 = std::make_tuple(1, 2, 3, 4);
  auto tup2 = std::make_tuple(33.1414, 1UL, 2, 22, 1.1, 3.141);

  //   std::cout << "We can print an entire tuple in one go, too:\n" <<
  //   std::endl; std::cout << tupletools::to_string(tup1) << "\n" <<
  //   std::endl;
  //   std::cout << tupletools::to_string(tup2) << "\n" << std::endl;
}

void
any_tests()
{
  auto tup1 = std::make_tuple(1, 2, 3, 4);
  auto tup2 = std::make_tuple(1, 2, 7, 4);

  auto ilist =
    tupletools::where([](auto x, auto y) { return x == y; }, tup1, tup2);

  bool b1 = tupletools::disjunction_of(ilist);
  //   std::cout << std::boolalpha;
  //   std::cout << b1 << std::endl;
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

void
range_tests()
{
  {
    int stop = -999999;
    int j = stop;
    auto _range = itertools::range(stop);
    for (auto i : _range) {
      assert(i == j);
      j++;
    }
    assert(j == 0);
  }
  {
    int stop = 999999;
    int j = 0;
    auto _range = itertools::range(stop);
    for (auto i : _range) {
      assert(i == j);
      j++;
    }
    assert(j == stop);
  }

  {
    int stop = -999999;
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
  if (n < 10) {
    co_yield rec(n + 1);
  } else {
    co_yield 99;
  }
  co_return;
};

int
main()
{

  any_tests();
  tupletools_zip_tests();
//   enumerate_tests();
//   range_tests();

  std::vector<int> v1(10, 10);

  for (auto [i, j] : itertools::zip(v1, std::vector<int>{1, 2, 3, 4})) {}

  //   for (auto [n, i] : itertools::enumerate(v1)) { fmt::print("{}, {}\n", n,
  //   i); }

  // //   for (auto& [n, i] : itertools::enumerate(v1)) {
  // //     fmt::print("{}, {}\n", n, i);
  // //     i += 1;
  // //   }
  //   for (const auto& [n, i] : itertools::enumerate(v1)) {
  //     fmt::print("{}, {}\n", n, i);
  //     // i += 1;
  //   }

  //   for (auto [n, i] : itertools::enumerate(v1)) { fmt::print("{}, {}\n", n,
  //   i); }

  return 0;
}