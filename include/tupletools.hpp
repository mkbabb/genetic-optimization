#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <list>
#include <numeric>
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
apply_impl(F func, std::index_sequence<Ix...>)
{
  return func(std::integral_constant<size_t, Ix>{}...);
}

template<size_t N, class F>
constexpr auto
apply(F func)
{
  return apply_impl(func, std::make_index_sequence<N>{});
}

template<class Tuple>
constexpr auto
reverse(Tuple tup)
{
  return apply<std::tuple_size<Tuple>{}>([&tup](auto... Ix) {
    return std::make_tuple(
      std::get<std::tuple_size<Tuple>{} - (Ix + 1)>(tup)...);
  });
}

template<class... Tuples>
constexpr auto
transpose(Tuples... tup)
{
  constexpr size_t len = std::min({ std::tuple_size<Tuples>{}... });
  auto row = [&](auto Ix) { return std::make_tuple(std::get<Ix>(tup)...); };
  return apply<len>([&](auto... Ixs) { return std::make_tuple(row(Ixs)...); });
}

template<class... Ts>
constexpr auto
deref(std::tuple<Ts...>& tup)
{
  return apply<sizeof...(Ts)>([&tup](auto... Ixs) {
    return std::forward_as_tuple(*std::get<Ixs>(tup)...);
  });
}

template<class... Ts>
constexpr auto
deref(const std::tuple<Ts...>& tup)
{
  return apply<sizeof...(Ts)>([&tup](auto... Ixs) {
    return std::forward_as_tuple(*std::get<Ixs>(tup)...);
  });
}

template<class... Ts>
constexpr auto
increment_ref(std::tuple<Ts...>& tup)
{
  return apply<sizeof...(Ts)>([&tup](auto... Ixs) {
    (void)std::initializer_list<int>{ [&tup, &Ixs] {
      ++std::get<Ixs>(tup);
      return 0;
    }()... };
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
  return apply<sizeof...(Ts)>([&](auto... Ixs) {
    const auto ilist = std::initializer_list<bool>{ [&] {
      return pred(std::get<Ixs>(tup1), std::get<Ixs>(tup2));
    }()... };
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
    if (i) {
      return true;
    }
  };
  return false;
}

template<class T>
constexpr bool
all_of(std::initializer_list<T> ilist)
{
  for (const auto& i : ilist) {
    if (!i) {
      return false;
    }
  };
  return true;
}

template<class T>
constexpr bool
disjunction_of(std::initializer_list<T> ilist)
{
  bool prev = true;
  for (auto& i : ilist) {
    if (!prev && i) {
      return false;
    };
    prev = i;
  };
  return true;
}

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
    using pointer = T*;
    using reference = T&;

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

template<class... Ts>
class zip
{
  static_assert(sizeof...(Ts) > 0, "!");

public:
  class iterator
  {
  protected:
    std::tuple<typename Ts::iterator...> seq_;

  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = std::tuple<typename Ts::value_type...>;
    using difference_type = std::tuple<typename Ts::value_type...>;
    using pointer = std::tuple<typename Ts::value_type*...>;
    using reference = std::tuple<typename Ts::value_type&...>;

    explicit iterator(typename Ts::iterator... seq)
      : seq_(seq...)
    {}

    iterator& operator++()
    {
      increment_ref(seq_);
      return *this;
    }

    iterator operator++(int)
    {
      auto tup = *this;
      increment_ref(tup);
      return tup;
    }

    bool operator==(const iterator& rhs)
    {
      return !any_of(
        where([](auto x, auto y) { return x != y; }, seq_, rhs.seq_));
    }
    bool operator!=(const iterator& rhs)
    {
      return !any_of(
        where([](auto x, auto y) { return x == y; }, seq_, rhs.seq_));
    }

    auto operator*() -> iterator::reference { return deref(seq_); }
    auto operator*() const -> const iterator::value_type { return deref(seq_); }
  };

  explicit zip(Ts&... seq)
    : begin_(seq.begin()...)
    , end_(seq.end()...){};

  zip(const zip& seq) = default;

  zip& operator=(const zip& rhs) = default;

  zip<Ts...>::iterator begin() { return begin_; }
  zip<Ts...>::iterator end() { return end_; }

  zip<Ts...>::iterator begin() const { return begin_; }
  zip<Ts...>::iterator end() const { return end_; }

  zip<Ts...>::iterator cbegin() const { return begin_; }
  zip<Ts...>::iterator cend() const { return end_; }

  zip<Ts...>::iterator begin_;
  zip<Ts...>::iterator end_;
};

template<class T>
auto
enumerate(T& iterable)
{
  range range_(iterable.size());
  return zip(range_, iterable);
}
};
