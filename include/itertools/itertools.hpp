#ifndef ITERTOOLS_H
#define ITERTOOLS_H

#include "generator.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#pragma once

namespace tupletools {

/*
A collection of compile-time tools for manipulating and tractalating std::tuple.
Tuple is an obstinate type: though it has been realtively surmounted in this
sense; many higher order abstractions are provided hereinafter.
 */

/*
Template-type meta-programming.
Used for either getting or setting types withal.
 */

template<class T>
struct remove_cvref
{
  typedef std::remove_cv_t<std::remove_reference_t<T>> type;
};

template<class T>
using remove_cvref_t = typename remove_cvref<T>::type;

template<class T>
struct is_tuple : std::false_type
{};
template<class... T>
struct is_tuple<std::tuple<T...>> : std::true_type
{};
template<class T>
constexpr bool is_tuple_v = is_tuple<remove_cvref_t<T>>::value;

template<typename T>
struct remove_cref
{
  using type = typename std::add_lvalue_reference_t<
    std::remove_const_t<std::remove_reference_t<T>>>;
};

template<class T>
using remove_cref_t = typename tupletools::remove_cref<T>::type;

template<typename T>
struct add_cref
{
  using type = typename std::add_lvalue_reference_t<
    std::add_const_t<std::remove_reference_t<T>>>;
};

template<class T>
using add_cref_t = typename tupletools::add_cref<T>::type;

// const down-and-up-cast: returns "value" with const either removed or added.

template<typename T>
constexpr auto&&
const_downcast(T&& value)
{
  using downcasted = tupletools::remove_cref_t<T>;
  return std::forward<downcasted>(const_cast<downcasted>(value));
}

template<typename T>
constexpr auto&&
const_upcast(T&& value)
{
  using upcasted = tupletools::add_cref_t<T>;
  return std::forward<upcasted>(const_cast<upcasted>(value));
}

// determinately grabs a container's inner type. Think of it as a super_decay_t.

template<typename T>
struct container_iterator_value
{
  using type = typename std::iterator_traits<
    std::decay_t<decltype(std::declval<T&>().begin())>>::value_type;
};
template<typename T>
using container_iterator_value_t = typename container_iterator_value<T>::type;

template<typename T>
struct container_iterator_type
{
  using type = decltype(std::declval<T>().begin());
};

template<typename T>
using container_iterator_type_t = typename container_iterator_type<T>::type;

template<std::size_t... Ns>
struct index_sequence
{};

template<typename T>
struct tuple_size
{
  const static std::size_t value =
    std::tuple_size<tupletools::remove_cvref_t<T>>::value;
};

template<typename T>
constexpr std::size_t tuple_size_v = tupletools::tuple_size<T>::value;

template<class T, class... Ts>
struct is_any : std::bool_constant<(std::is_same_v<T, Ts> || ...)>
{};

template<class T, class... Ts>
struct is_all : std::bool_constant<(std::is_same_v<T, Ts> && ...)>
{};

template<template<typename, typename> class Pred, typename... Ts>
struct any_of_t : std::false_type
{};

template<template<typename, typename> class Pred,
         typename T0,
         typename T1,
         typename... Ts>
struct any_of_t<Pred, T0, T1, Ts...>
  : std::integral_constant<bool,
                           Pred<T0, T1>::value ||
                             any_of_t<Pred, T0, Ts...>::value>
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

// template-type meta-programming.

/*
The actual tupletools:
 */

// The below three functions are drivers behind one's basic argument forwarding
// of a tuple.

template<std::size_t... Ixs, class Func>
constexpr auto
index_apply_impl(Func&& func, std::index_sequence<Ixs...>)
{
  return func(std::integral_constant<size_t, Ixs>{}...);
}

template<size_t N, class Func>
constexpr auto
index_apply(Func&& func)
{
  return index_apply_impl(std::forward<Func>(func),
                          std::make_index_sequence<N>{});
}

/*
Essentially a clone of std::apply; an academic exercise.
 */
template<class Tup, class Func, const std::size_t N = tuple_size<Tup>::value>
constexpr auto
apply(Tup&& tup, Func&& func)
{
  return index_apply<N>([&](auto... Ixs) {
    return func(std::get<Ixs>(std::forward<Tup>(tup))...);
  });
}

/*
Begets a tuple of size N of type T, filled with value value.

@param value: fill value

@returns: tuple filled N times with value.
 */
template<const std::size_t N, class T>
constexpr auto
make_tuple_of(T value)
{
  auto func = [&value](auto t) { return value; };
  auto tup =
    index_apply<N>([&value](auto... Ix) { return std::make_tuple(Ix...); });
  return index_apply<N>(
    [&](auto... Ixs) { return std::make_tuple(func(std::get<Ixs>(tup))...); });
}

/*
High order functor that allows for a functor of type "Func" to be applied
element-wise to a tuple of type "Tup". "func" must accept two parameters of
types (size_t, tup_i).

Each item within "tup" is iterated upon, and must therefor each element must be
of a coalesceable type; the aforesaid elements must be of a type that allows
proper type coalescence.

@param tup: tuple of coalesceable type.
@param func:    high order function that takes two arguments of types
                (size_t,tup_i), wherein "tup_i" is a coalescably-typed element
                of "tup"

@returns: string representation of tup.
 */

template<class Tup, class Func, const std::size_t N = tuple_size_v<Tup>>
void
for_each(Tup&& tup, Func&& func)
{
  index_apply<N>([&](auto... Ix) {
    (func(Ix,
          std::forward<tupletools::remove_cvref_t<decltype(std::get<Ix>(tup))>>(
            std::get<Ix>(tup))),
     ...);
  });
  return;
}

template<class Tup, const std::size_t N = tuple_size<Tup>::value>
constexpr auto
reverse(Tup&& tup)
{
  return index_apply<N>([&tup](auto... Ix) {
    return std::make_tuple(std::get<N - (Ix + 1)>(tup)...);
  });
}

template<class... Tuples,
         const std::size_t N = std::min({std::tuple_size<Tuples>{}...})>
constexpr auto
transpose(Tuples&&... tups)
{
  auto row = [&](auto Ix) { return std::make_tuple(std::get<Ix>(tups)...); };
  return index_apply<N>(
    [&](auto... Ixs) { return std::make_tuple(row(Ixs)...); });
}

template<class Tup, const std::size_t N = tuple_size<Tup>::value>
constexpr auto
deref_fwd_const(Tup&& tup)
{
  return index_apply<N>([&tup](auto... Ixs) {
    return std::forward_as_tuple(*std::get<Ixs>(tup)...);
  });
}

template<class Tup, const std::size_t N = tuple_size<Tup>::value>
constexpr auto
deref_fwd_volatile(Tup&& tup)
{
  return index_apply<N>([&tup](auto... Ixs) {
    return std::forward_as_tuple(
      const_cast<tupletools::remove_cref_t<decltype(*std::get<Ixs>(tup))>>(
        *std::get<Ixs>(tup))...);
  });
}

template<class Tup, const std::size_t N = tuple_size<Tup>::value>
constexpr auto
deref_copy(Tup&& tup)
{
  return index_apply<N>(
    [&tup](auto... Ixs) { return std::make_tuple(*std::get<Ixs>(tup)...); });
}

template<class Tup>
constexpr auto
increment_ref(Tup&& tup)
{
  return tupletools::for_each(std::forward<std::decay_t<Tup>>(tup),
                              [](auto&& n, auto&& v) { v++; });
}

template<class Pred,
         class Tup1,
         class Tup2,
         const std::size_t N = tupletools::tuple_size_v<Tup1>,
         const std::size_t M = tupletools::tuple_size_v<Tup2>>
constexpr auto
where(Pred&& pred, Tup1&& tup1, Tup2&& tup2)
{
  static_assert(N == M, "Tuples must be the same size!");

  return index_apply<N>([&](auto... Ixs) {
    auto tup =
      std::make_tuple(pred(std::get<Ixs>(tup1), std::get<Ixs>(tup2))...);
    return tup;
  });
}

/*
Returns the string format representation of a coalesceably-typed tuple;
The elements of aforesaid tuple must be of a type that allows proper
type coalescence.

@param tup: tuple of coalesceable type.
@returns: string representation of tup.
 */
template<typename Tup, const std::size_t N = tuple_size<Tup>::value>
std::string
to_string(Tup&& tup)
{
  std::string s = "(";

  tupletools::for_each(
    std::forward<tupletools::remove_cvref_t<Tup>>(tup), // Not sure why decay_t
                                                        // is needed here...
    [&tup, &s](auto&& n, auto&& v) {
      s += std::to_string(v);
      s += n < N - 1 ? ", " : "";
      return false;
    });
  return s + ")";
}

template<class Tup>
constexpr bool
any_of(Tup&& tup)
{
  bool b = false;
  auto func = [&](auto&& n, auto&& v) -> bool {
    if (v) {
      b = true;
      return true;
    }
    return false;
  };
  tupletools::for_each(std::forward<Tup>(tup),
                       std::forward<decltype(func)>(func));
  return b;
}

template<class Tup>
constexpr bool
all_of(Tup&& tup)
{
  bool b = true;
  auto func = [&](auto&& n, auto&& v) -> bool {
    if (!v) {
      b = false;
      return true;
    }
    return false;
  };
  tupletools::for_each(std::forward<Tup>(tup),
                       std::forward<decltype(func)>(func));
  return b;
}

template<class Tup>
constexpr bool
disjunction_of(Tup&& tup)
{
  bool b = true;
  bool prev_b = true;
  auto func = [&](auto&& n, auto&& v) -> bool {
    if (!prev_b && v) {
      b = false;
      return true;
    } else {
      prev_b = v;
      return false;
    }
  };
  tupletools::for_each(std::forward<Tup>(tup),
                       std::forward<decltype(func)>(func));
  return b;
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

template<class Tup, std::enable_if_t<!is_tuple_v<Tup>, int> = 0>
constexpr auto
make_tuple_if(Tup&& tup)
{
  return std::make_tuple(tup);
}

template<class Tup, std::enable_if_t<is_tuple_v<Tup>, int> = 0>
constexpr auto
make_tuple_if(Tup&& tup)
{
  return tup;
}

template<class... T>
struct flatten_impl
{};

template<class T>
struct flatten_impl<T>
{
  template<class U>
  constexpr auto operator()(U&& value)
  {
    return std::forward<U>(value);
  }
};

template<class T>
struct flatten_impl<std::tuple<T>>
{
  template<class Tup>
  constexpr auto operator()(Tup&& tup)
  {
    return flatten_impl<tupletools::remove_cvref_t<T>>{}(std::get<0>(tup));
  }
};

template<class T, class... Ts>
struct flatten_impl<std::tuple<T, Ts...>>
{
  template<class Tup,
           const std::size_t N = sizeof...(Ts),
           std::enable_if_t<(N >= 1), int> = 0>
  constexpr auto operator()(Tup&& tup)
  {
    auto tup_first =
      flatten_impl<tupletools::remove_cvref_t<T>>{}(std::get<0>(tup));

    auto t_tup_args = index_apply<N>([&tup](auto... Ixs) {
      return std::make_tuple(std::get<Ixs + 1>(tup)...);
    });
    auto tup_args =
      flatten_impl<tupletools::remove_cvref_t<decltype(t_tup_args)>>{}(
        t_tup_args);

    return std::tuple_cat(make_tuple_if(tup_first), make_tuple_if(tup_args));
  }
};

template<class Tup>
constexpr auto
flatten(Tup&& tup)
{
  return flatten_impl<tupletools::remove_cvref_t<Tup>>{}(
    std::forward<Tup>(tup));
}

};

namespace itertools {

using namespace tupletools;

/*
The zip and zip-iterator classes, respectively.

Allows one to "zip" determinately any iterable type together, yielding thereupon
subsequent iterations an n-tuple of respective iterable values.

Each value yielded by aforesaid zip is declared as "const & T", so no copies are
every made or attempted withal. The obstinancy of the yielded type is due to the
rvalue acceptance provided herein. If a volatile reference is sought, one may
circumvent the type by using the aforeprovided "const_downcast" methoed
wherewith one may trivially assign or manipulate a now volatile reference.

An example usage of zip:

    std::vector<int> v1(10, 10);
    std::list<float> l1(7, 1.234);

    for (auto [i, j] : itertools::zip(v1, v2)) {
        ...
    }

And to modify the const & of i or j:

    tupletools::const_downcast(i) = "your value";

An example of zip's r-value acquiescence:

    std::vector<int> v1(10, 10);

    for (auto [i, j] : itertools::zip(v1, std::list<float>{1.2, 1.2, 1.2}))
    {
        ...
    }

Notice that the hereinbefore shown containers were of unequal length:
the zip iterator automatically scales downward to the smallest of the given
container sizes.
 */

template<class... Args>
class zip;

template<class... Args>
class zip_iterator
  : public std::iterator<std::forward_iterator_tag,
                         container_iterator_type_t<Args&>...>
{
public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = std::tuple<container_iterator_value_t<Args>...>;
  using pointer_type = std::tuple<container_iterator_value_t<Args>*...>*;
  using reference_type = std::tuple<container_iterator_value_t<Args&>&...>&;

  explicit zip_iterator(container_iterator_type_t<Args&>... args) noexcept
    : _args(args...)
  {}

  zip_iterator() noexcept = default;

  zip_iterator& operator++()
  {
    increment_ref(_args);
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
    return any_of(
      where([](auto x, auto y) { return x == y; }, _args, rhs._args));
  }
  bool operator!=(const zip_iterator& rhs) noexcept { return !(*this == rhs); }

  const auto operator*() noexcept { return deref_fwd_const(_args); }
  auto operator-> () const noexcept -> pointer_type { return _args; }

private:
  std::tuple<container_iterator_type_t<Args&>...> _args;
};

/*

Zips an arbitrary number of iterables together into one iterable container. Each
iterable herein must provide begin and end member functions (whereof are used to
iterate upon each iterable within the container).

There are two constructors of zip:

@param args: of type const Args&...

and

@param args: of type const std::tuple<Args...>&

Meaning, one can provide either a packed or unpacked tuple of iterables.
 */
template<class... Args>
class zip
{
  static constexpr std::size_t N = sizeof...(Args);
  static_assert(N > 0, "!");

public:
  using iterator = zip_iterator<const Args&...>;

  template<std::enable_if_t<!(tupletools::is_tuple_v<Args> && ...), int> = 0>
  zip(const Args&... args)
    : _begin(args.begin()...)
    , _end(args.end()...){};

  zip(const std::tuple<Args...>& args)
  {
    auto func = [&](const auto&... _args) { return zip<Args...>(_args...); };
    auto zppd = tupletools::apply(std::forward<decltype(args)>(args),
                                  std::forward<decltype(func)>(func));
    _begin = zppd._begin;
    _end = zppd._end;
  }

  zip& operator=(const zip& rhs) = default;

  iterator begin() { return _begin; }
  iterator end() { return _end; }

private:
  iterator _begin;
  iterator _end;
};

/*

Coroutine generator function. Works with both negative and positive stride
values.

@param start: T starting value.
@param stop: T stopping value.
@param stride: T stride value whereof start is incremented.

@co_yields: incremented starting value of type T.
 */

template<typename T = std::size_t>
generator<T>
range(T start, T stop, T stride = 1)
{
  stride = start > stop ? -1 : 1;
  do {
    co_yield start;
    start += stride;
  } while (start < stop);
}

template<typename T = std::size_t>
generator<T>
range(T stop)
{
  T start = 0;
  if (start > stop) { std::swap(start, stop); }
  return range<T>(start, stop, 1);
}

template<class Iterable>
constexpr auto
enumerate(Iterable&& iter)
{
  auto _range = range<std::size_t>(iter.size());
  return zip(_range, std::forward<Iterable>(iter));
}

template<class Iterable>
constexpr Iterable
swap(Iterable&& iter, int ix1, int ix2)
{
  auto t = iter[ix1];
  iter[ix1] = iter[ix2];
  iter[ix2] = t;
  return iter;
}

/*
High order functor, whereof: applies a binary function of type BinaryFunction,
func, over an iterable range of type Iterable, iter. func must accept two values
of (std::size_t, IterableValue), wherein IterableValue is the iterable's
container value. func must also return a boolean value upon each iteration:
returning true, subsequently forcing a break in the loop, or false to continue
onward.

@param iter: iterable function of type Iterable.
@param func: low order functor of type BinaryFunction; returns a boolean.

@returns iter
 */
template<class Iterable,
         class BinaryFunction,
         class IterableValue = tupletools::container_iterator_value_t<Iterable>>
constexpr Iterable
for_each(Iterable&& iter, BinaryFunction&& func)
{
  for (const auto& [n, i] : enumerate(iter)) {
    bool b =
      func(std::forward<std::size_t>(n), std::forward<const IterableValue>(i));
    if (b) { break; }
  }
  return iter;
}

template<class Iterable>
constexpr void
roll(Iterable&& iter, int axis)
{
  int ndim = iter.size();
  if (axis == 0) {
    return;
  } else if (axis < 0) {
    axis += ndim;
  }
  int i = 0;
  while (i++ < axis) { swap(std::forward<Iterable>(iter), axis, i); };
}

template<typename Iterable>
std::string
join(Iterable&& iter, std::string const&& sep)
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
