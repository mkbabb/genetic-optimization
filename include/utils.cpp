#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <vector>

constexpr uint32_t DB_32 = 0x4653ADF;
constexpr uint64_t DB_64 = 0x07EDD5E59A4E28C2;
constexpr uint32_t TAB_32[32] = { 0,  1,  2,  6,  3,  11, 7,  16, 4,  14, 12,
                                  21, 8,  23, 17, 26, 31, 5,  10, 15, 13, 20,
                                  22, 25, 30, 9,  19, 24, 29, 18, 28, 27 };

constexpr uint64_t TAB_64[64] = { 63, 0,  58, 1,  59, 47, 53, 2,  60, 39, 48,
                                  27, 54, 33, 42, 3,  61, 51, 37, 40, 49, 18,
                                  28, 20, 55, 30, 34, 11, 43, 14, 22, 4,  62,
                                  57, 46, 52, 38, 26, 32, 41, 50, 36, 17, 19,
                                  29, 10, 13, 21, 56, 45, 25, 31, 35, 16, 9,
                                  12, 44, 24, 15, 8,  23, 7,  6,  5 };

int
ilog2(uint32_t x)
{
  return TAB_32[((x & -x) * DB_32) >> 27];
};

int
ilog2(int x)
{
  return TAB_64[((x & -x) * DB_64) >> 58];
};

template<typename Iter>
std::string
join(Iter begin, Iter end, std::string const& sep)
{
  std::ostringstream result;
  if (begin != end)
    result << *begin++;
  while (begin != end)
    result << sep << *begin++;
  return result.str();
}

namespace ranges {
template<typename Range, typename Function>
Function
for_each(Range& range, Function f)
{
  return std::for_each(begin(range), end(range), f);
};

template<typename Range>
decltype(auto)
join(Range& range, std::string const& sep)
{
  return ::join(begin(range), end(range), sep);
}
}

template<typename Func>
struct y_combinator
{
  Func func;
  y_combinator(Func func)
    : func(std::move(func))
  {}
  template<typename... Args>
  decltype(auto) operator()(Args&&... args) const
  {
    return func(std::ref(*this), std::forward<Args>(args)...);
  }
  template<typename... Args>
  decltype(auto) operator()(Args&&... args)
  {
    return func(std::ref(*this), std::forward<Args>(args)...);
  }
};

// Python's implementation of __round__ and double_round
template<typename T>
auto
double_round(T x, int ndigits = 0) -> double
{
  double pow1, pow2, y, z;
  if (ndigits >= 0) {
    if (ndigits > 22) {
      /* pow1 and pow2 are each safe from overflow, but
         pow1*pow2 ~= pow(10.0, ndigits) might overflow */
      pow1 = pow(10.0, (double)(ndigits - 22));
      pow2 = 1e22;
    } else {
      pow1 = pow(10.0, (double)ndigits);
      pow2 = 1.0;
    }
    y = (x * pow1) * pow2;
    /* if y overflows, then rounded value is exactly x */
  } else {
    pow1 = pow(10.0, (double)-ndigits);
    pow2 = 1.0; /* unused; silences a gcc compiler warning */
    y = x / pow1;
  }

  z = round(y);
  if (fabs(y - z) == 0.5)
    /* halfway between two integers; use round-half-even */
    z = 2.0 * round(y / 2.0);

  if (ndigits >= 0)
    z = (z / pow2) / pow1;
  else
    z *= pow1;
  return z;
}

auto
de_bruijn(int k, int n) -> std::string
{
  std::vector<int> a(k * n, 0);
  std::vector<int> seq;

  auto db = y_combinator([&](auto self, int t, int p) -> void {
    if (t > n) {
      if (!(n % p)) {
        seq.insert(end(seq), begin(a) + 1, begin(a) + p + 1);
      }
    } else {
      a[t] = a[t - p];
      self(t + 1, p);
      for (int j = a[t - p] + 1; j < k; j++) {
        a[t] = j;
        self(t + 1, t);
      }
    }
  });
  db(1, 1);
  return ranges::join(seq, "");
}

auto
tabn(int n, int de_bruijn_n) -> std::vector<int>
{
  int lgn = log2(n);
  std::vector<int> v(n, 0);
  for (int i = 0; i < n; i++) {
    int ix = (de_bruijn_n << i) >> (n - lgn);
    ix &= (n - 1);
    v[ix] = i;
  }
  return v;
}
