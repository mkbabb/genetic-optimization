#define FMT_HEADER_ONLY

#include "fmt/format.h"
#include "itertools/generator.hpp"
#include "itertools/itertools.hpp"
#include "random_t/random_t.hpp"
#include "utils.cpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iterator>
#include <list>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

void
join_tests()
{
  std::vector<std::string> v1 = {"h", "e", "l", "l", "o"};
  std::vector<int> v2 = {1, 2, 3, 4, 5, 6, 7, 8};

  assert(itertools::join(v1, "") == "hello");
  assert(itertools::join(v1, ",") == "h,e,l,l,o");
  assert(itertools::join(v2, ", ") == "1, 2, 3, 4, 5, 6, 7, 8");
  fmt::print("join tests complete");
}

void
ilog_tests()
{
  int m = 5;
  int tm = std::pow(2, m);

  std::string db = de_bruijn(2, m);
  std::vector<int> tb = tabn(tm, std::stoi(db, nullptr, 2));}

int
main()
{
  join_tests();
  ilog_tests();

  return 1;
}