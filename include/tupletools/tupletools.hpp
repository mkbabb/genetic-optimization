#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <list>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#pragma once

// template <std::size_t... Ns>
// struct index_sequence {};

// template <std::size_t N, std::size_t... Is>
// auto make_index_sequence_impl() {
//     if constexpr (N == 0) {
//         return index_sequence<Is...>();
//     } else {
//         return make_index_sequence_impl<N - 1, N - 1, Is...>();
//     };
// }

// template <std::size_t N>
// using make_index_sequence =
//     std::decay_t<decltype(make_index_sequence_impl<N>())>;

namespace tupletools {
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

template<const std::size_t N, const std::size_t M, class... Ts, class Func>
constexpr auto
foreach_impl(std::tuple<Ts...>& tup, Func func)
{
  if constexpr (N == M) {
    return;
  } else {
    auto& val = std::get<N>(tup);
    func(val, N);
    return foreach_impl<N + 1, M>(tup, func);
  };
};

template<class... Ts, class Func>
void foreach (std::tuple<Ts...>& tup, Func func)
{
  foreach_impl<0, sizeof...(Ts)>(tup, func);
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

template<class... Ts>
constexpr auto
deref(std::tuple<Ts...>& tup)
{
  return index_apply<sizeof...(Ts)>([&tup](auto... Ixs) {
    return std::forward_as_tuple(*std::get<Ixs>(tup)...);
  });
}

template<class... Ts>
constexpr auto
deref(const std::tuple<Ts...>& tup)
{
  return index_apply<sizeof...(Ts)>([&tup](auto... Ixs) {
    return std::forward_as_tuple(*std::get<Ixs>(tup)...);
  });
}

template<class... Ts>
constexpr auto
increment_ref(std::tuple<Ts...>& tup)
{
  return index_apply<sizeof...(Ts)>([&tup](auto... Ixs) {
    (void) std::initializer_list<int>{[&tup, &Ixs] {
      ++std::get<Ixs>(tup);
      return 0;
    }()...};
  });
}

template<class P, class... Ts, class... Us>
constexpr auto
where(const P& pred,
      const std::tuple<Ts...>& tup1,
      const std::tuple<Us...>& tup2)
{
  static_assert(sizeof...(Ts) == sizeof...(Us),
                "Tuples must be the same "
                "size!");
  return index_apply<sizeof...(Ts)>([&](auto... Ixs) {
    const auto ilist = std::initializer_list<bool>{
      [&] { return pred(std::get<Ixs>(tup1), std::get<Ixs>(tup2)); }()...};
    return ilist;
  });
}

template<std::size_t N, std::size_t M, class... Ts>
constexpr auto
to_string_impl(const std::tuple<Ts...>& tup, std::string& s)
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

template<class... Ts>
std::string
to_string(const std::tuple<Ts...>& tup)
{
  std::string s = "(";
  to_string_impl<0, sizeof...(Ts)>(tup, s);
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

template<class... Ts>
class zip;

template<class... Ts>
class zip_iterator
{
public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = std::tuple<typename iterator_value<Ts>::type...>;
  using pointer_type = std::tuple<typename iterator_value<Ts>::type*...>;
  using reference_type = std::tuple<typename iterator_value<Ts>::type&...>;

  explicit zip_iterator(decltype(std::declval<Ts&>().begin())... seq) noexcept
    : seq_(seq...)
  {}

  zip_iterator() = default;

  zip_iterator& operator++()
  {
    tupletools::increment_ref(seq_);
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
    return !tupletools::any_of(
      tupletools::where([](auto x, auto y) { return x != y; }, seq_, rhs.seq_));
  }
  bool operator!=(const zip_iterator& rhs) noexcept
  {
    return !tupletools::any_of(
      tupletools::where([](auto x, auto y) { return x == y; }, seq_, rhs.seq_));
  }

  auto operator*() noexcept -> reference_type
  {
    return tupletools::deref(seq_);
  }
  auto operator*() const noexcept -> const value_type
  {
    return tupletools::deref(seq_);
  }
  auto operator-> () const noexcept -> pointer_type { return seq_; }

private:
  std::tuple<decltype(std::declval<Ts&>().begin())...> seq_;
};

template<class... Ts>
class zip
{
public:
  static_assert(sizeof...(Ts) > 0, "!");

  using iterator = zip_iterator<Ts...>;

  explicit zip(Ts&... seq)
    : _begin(
        std::forward<decltype(std::declval<Ts&>().begin())>(seq.begin())...)
    , _end(std::forward<decltype(std::declval<Ts&>().begin())>(seq.end())...){};

  zip(const zip& seq) = default;

  zip& operator=(const zip& rhs) = default;

  iterator begin() { return _begin; }
  iterator end() { return _end; }

  iterator begin() const { return _begin; }
  iterator end() const { return _end; }

  iterator cbegin() const { return _begin; }
  iterator cend() const { return _end; }

private:
  friend class zip_iterator<Ts...>;
  iterator _begin;
  iterator _end;
};

template<class T>
class range
{
public:
  using value_type = T;
  class iterator
  {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = T;
    using difference_type = T;
    using pointer_type = T*;
    using reference_type = T&;

    explicit iterator(range<T>& seq)
      : seq_(seq)
    {}

    iterator& operator++()
    {
      seq_.current_ += seq_.stride_;
      return *this;
    }

    iterator operator++(int)
    {
      auto tmp = *this;
      tmp.seq_.current_ += tmp.seq_.stride_;
      return tmp;
    }

    bool operator!=(const iterator& rhs)
    {
      return seq_.current_ != rhs.seq_.current_;
    }

    bool operator==(const iterator& rhs)
    {
      return seq_.current_ == rhs.seq_.current_;
    }

    auto operator*() -> T& { return seq_.current_; }

  protected:
    range<T> seq_;
  };

  explicit range(T last)
    : first_(0)
    , last_(last)
    , stride_(1)
  {
    current_ = first_;
    size_ = std::abs(static_cast<int>(last_));
  };

  range(T first, T last)
    : first_(first)
    , last_(last)
  {
    current_ = first_;
    if (first_ > last_) {
      stride_ = -1;
    } else if (first_ <= last_) {
      stride_ = 1;
    }
    size_ = std::abs(static_cast<int>(last_ - first_));
  }

  range(T first, T last, T stride)
    : first_(first)
    , last_(last)
    , stride_(stride)
  {
    current_ = first_;
    size_ = std::abs(static_cast<int>(last_ - first_));
  }

  auto first() const -> const T& { return first_; }
  auto first() -> T& { return first_; }

  auto last() const -> const T& { return last_; }
  auto last() -> T& { return last_; }

  auto stride() const -> const T& { return stride_; }
  auto stride() -> T& { return stride_; }

  auto current() const -> const T& { return current_; }
  auto current() -> T& { return current_; }

  int size() const { return size_; }

  range<T>::iterator begin() { return iterator(*this); }

  range<T>::iterator end()
  {
    range tmp(0);
    tmp.current() = last_;
    return iterator(tmp);
  }

protected:
  T first_, last_, current_, stride_;
  int size_;
};

template<class T>
constexpr auto
enumerate(T& iterable)
{
  range range_(iterable.size());
  return zip(range_, iterable);
}

template<class T>
constexpr void
swap(T& cont, int ix1, int ix2)
{
  auto t = cont[ix1];
  cont[ix1] = cont[ix2];
  cont[ix2] = t;
}

template<class InputCont, class UnaryFunction>
constexpr InputCont
for_each(InputCont& cont, UnaryFunction f)
{
  auto first = begin(cont);
  auto last = end(cont);
  for (int n = 0; first != last; ++first) { f(*first, n++); }
  return cont;
}

template<class InputCont>
constexpr void
roll(InputCont cont, int axis)
{
  int ndim = cont.size();
  if (axis == 0) {
    return;
  } else if (axis < 0) {
    axis += ndim;
  }
  int i = 0;
  while (i++ < axis) { swap(cont, axis, i); };
}
}
