#ifndef ITERTOOLS_H
#define ITERTOOLS_H

#include "generator.hpp"

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

#pragma once

namespace tupletools {

template<std::size_t... Ns>
struct index_sequence
{};

template<std::size_t N, std::size_t... Is>
auto
make_index_sequence_impl()
{
  if constexpr (N == 0) {
    return index_sequence<Is...>();
  } else {
    return make_index_sequence_impl<N - 1, N - 1, Is...>();
  };
}

template<std::size_t N>
using make_index_sequence =
  std::decay_t<decltype(make_index_sequence_impl<N>())>;

template<std::size_t... Ix, class F>
constexpr auto
index_apply_impl(F func, std::index_sequence<Ix...>)
{
  return func(std::integral_constant<size_t, Ix>{}...);
}

template<size_t N, class F>
constexpr auto
index_apply(F func)
{
  return index_apply_impl(func, std::make_index_sequence<N>{});
}

template<class Tuple, class F>
constexpr auto
apply(Tuple t, F f)
{
  return index_apply<std::tuple_size<Tuple>{}>(
    [&](auto... Is) { return f(std::get<Is>(t)...); });
}

template<const std::size_t N, const std::size_t M, class... Args, class Func>
constexpr auto
foreach_impl(std::tuple<Args...>& tup, Func func)
{
  if constexpr (N == M) {
    return;
  } else {
    auto& val = std::get<N>(tup);
    func(val, N);
    return foreach_impl<N + 1, M>(tup, func);
  };
};

template<class... Args, class Func>
void foreach (std::tuple<Args...>& tup, Func func)
{
  foreach_impl<0, sizeof...(Args)>(tup, func);
  return;
}

template<class Tuple>
constexpr auto
reverse(Tuple tup)
{
  return index_apply<std::tuple_size<Tuple>{}>([&tup](auto... Ix) {
    return std::make_tuple(
      std::get<std::tuple_size<Tuple>{} - (Ix + 1)>(tup)...);
  });
}

template<class... Tuples>
constexpr auto
transpose(Tuples... tup)
{
  constexpr size_t len = std::min({std::tuple_size<Tuples>{}...});
  auto row = [&](auto Ix) { return std::make_tuple(std::get<Ix>(tup)...); };
  return index_apply<len>(
    [&](auto... Ixs) { return std::make_tuple(row(Ixs)...); });
}

template<class... Args>
constexpr auto
deref(std::tuple<Args...>& tup)
{
  return index_apply<sizeof...(Args)>([&tup](auto... Ixs) {
    return std::forward_as_tuple(*std::get<Ixs>(tup)...);
  });
}

template<class... Args>
constexpr auto
deref(const std::tuple<Args...>& tup)
{
  return index_apply<sizeof...(Args)>([&tup](auto... Ixs) {
    return std::forward_as_tuple(*std::get<Ixs>(tup)...);
  });
}

template<class... Args>
constexpr auto
increment_ref(std::tuple<Args...>& tup)
{
  return index_apply<sizeof...(Args)>([&tup](auto... Ixs) {
    (void) std::initializer_list<int>{[&tup, &Ixs] {
      ++std::get<Ixs>(tup);
      return 0;
    }()...};
  });
}

template<class P, class... Args, class... Us>
constexpr auto
where(const P& pred,
      const std::tuple<Args...>& tup1,
      const std::tuple<Us...>& tup2)
{
  static_assert(sizeof...(Args) == sizeof...(Us),
                "Tuples must be the same size!");
  return index_apply<sizeof...(Args)>([&](auto... Ixs) {
    const auto ilist = std::initializer_list<bool>{
      [&] { return pred(std::get<Ixs>(tup1), std::get<Ixs>(tup2)); }()...};
    return ilist;
  });
}

template<std::size_t N, std::size_t M, class... Args>
constexpr auto
to_string_impl(const std::tuple<Args...>& tup, std::string& s)
{
  if constexpr (N == M) {
    s += ")";
    return s;
  } else {
    auto val = std::get<N>(tup);
    if (std::is_integral<decltype(val)>::value) {
      s += std::to_string(val);
    } else {
      s += val;
    }
    s += (N < M - 1) ? ", " : "";
    return to_string_impl<N + 1, M>(tup, s);
  };
};

template<class... Args>
std::string
to_string(const std::tuple<Args...>& tup)
{
  std::string s = "(";
  to_string_impl<0, sizeof...(Args)>(tup, s);
  return s;
}

template<class T>
constexpr bool
any_of(std::initializer_list<T> ilist)
{
  for (const auto& i : ilist) {
    if (i) { return true; }
  };
  return false;
}

template<class T>
constexpr bool
all_of(std::initializer_list<T> ilist)
{
  for (const auto& i : ilist) {
    if (!i) { return false; }
  };
  return true;
}

template<class T>
constexpr bool
disjunction_of(std::initializer_list<T> ilist)
{
  bool prev = true;
  for (auto& i : ilist) {
    if (!prev && i) { return false; };
    prev = i;
  };
  return true;
}
};

namespace itertools {

template<typename T>
struct iterator_value
{
  using type = typename std::iterator_traits<
    std::decay_t<decltype((std::declval<T&>().begin()))>>::value_type;
};

template<class... Args>
class zip;

template<class... Args>
class zip_iterator
{
public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = std::tuple<typename iterator_value<Args>::type...>;
  using pointer_type = std::tuple<typename iterator_value<Args>::type*...>;
  using reference_type = std::tuple<typename iterator_value<Args>::type&...>;

  explicit zip_iterator(
    decltype(std::declval<Args&>().begin())... args) noexcept
    : _args(args...)
  {}

  zip_iterator() = default;

  zip_iterator& operator++()
  {
    tupletools::increment_ref(_args);
    return *this;
  }

  zip_iterator operator++(int)
  {
    auto tup = *this;
    increment_ref(tup);
    return tup;
  }

  bool operator==(const zip_iterator& rhs) noexcept
  {
    return !tupletools::any_of(tupletools::where(
      [](auto x, auto y) { return x != y; }, _args, rhs._args));
  }
  bool operator!=(const zip_iterator& rhs) noexcept
  {
    return !tupletools::any_of(tupletools::where(
      [](auto x, auto y) { return x == y; }, _args, rhs._args));
  }

  auto operator*() noexcept -> reference_type
  {
    return tupletools::deref(_args);
  }
  auto operator*() const noexcept -> const value_type
  {
    return tupletools::deref(_args);
  }
  auto operator-> () const noexcept -> pointer_type { return _args; }

private:
  std::tuple<decltype(std::declval<Args&>().begin())...> _args;
};

template<class... Args>
class zip
{
public:
  static_assert(sizeof...(Args) > 0, "!");

  using iterator = zip_iterator<Args...>;

  explicit zip(Args&... args)
    : _begin(
        std::forward<decltype(std::declval<Args&>().begin())>(args.begin())...)
    , _end(
        std::forward<decltype(std::declval<Args&>().begin())>(args.end())...){};

  zip(const zip& args) = default;

  zip& operator=(const zip& rhs) = default;

  iterator begin() { return _begin; }
  iterator end() { return _end; }

  iterator begin() const { return _begin; }
  iterator end() const { return _end; }

  iterator cbegin() const { return _begin; }
  iterator cend() const { return _end; }

private:
  friend class zip_iterator<Args...>;
  iterator _begin;
  iterator _end;
};

template<typename T>
generator<T>
range(T start, T stop, T stride = 1)
{
  stride = start > stop ? -1 : 1;
  do {
    co_yield start;
    start += stride;
  } while (start < stop);
}

template<typename T>
generator<T>
range(T stop)
{
  T start = 0;
  if (start > stop) { std::swap(start, stop); }
  return range<T>(start, stop, 1);
}

template<class Iter>
constexpr auto
enumerate(Iter& iter)
{
  auto _range = range(iter.size());
  return zip(_range, iter);
}

template<class Iter>
constexpr void
swap(Iter& iter, int ix1, int ix2)
{
  auto t = iter[ix1];
  iter[ix1] = iter[ix2];
  iter[ix2] = t;
}

template<class Iter, class UnaryFunction>
constexpr Iter
for_each(Iter& iter, UnaryFunction f)
{
  for (auto [n, i] : enumerate(iter)) { f(n, i); }
  return iter;
}

template<class Iter>
constexpr void
roll(Iter& iter, int axis)
{
  int ndim = iter.size();
  if (axis == 0) {
    return;
  } else if (axis < 0) {
    axis += ndim;
  }
  int i = 0;
  while (i++ < axis) { swap(iter, axis, i); };
}

template<typename Iter>
std::string
join(Iter& iter, std::string const&& sep)
{
  std::ostringstream result;
  for (auto [n, i] : enumerate(iter)) { result << (n > 0 ? sep : "") << i; }
  return result.str();
}

template<typename Func>
struct y_combinator
{
  Func func;
  y_combinator(Func func)
    : func(std::move(func))
  {}
  template<typename... Args>
  auto operator()(Args&&... args) const
  {
    return func(std::ref(*this), std::forward<Args>(args)...);
  }
  template<typename... Args>
  auto operator()(Args&&... args)
  {
    return func(std::ref(*this), std::forward<Args>(args)...);
  }
};

};
#endif // ITERTOOLS_H
