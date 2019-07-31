#ifndef GENERATOR_H
#define GENERATOR_H

#include <exception>
#include <experimental/coroutine>
#include <iterator>
#include <type_traits>
#include <utility>

#pragma once

namespace itertools {

template<typename T>
class generator;

template<typename T>
class generator_iterator;

template<typename T>
class generator_promise;

template<typename T>
class awaitable
{
public:
  awaitable(generator_promise<T>* promise)
    : _promise(promise)
  {}

  bool await_ready() noexcept { return _promise == nullptr; }

  template<typename U>
  void await_suspend(U) noexcept
  {}

  void await_resume()
  {
    if (!_promise) { _promise->rethrow_if_exception(); }
  }

private:
  generator_promise<T>* _promise;
};

template<typename T>
class generator_promise
{
public:
  using value_type = std::remove_reference_t<T>;
  using reference_type = std::conditional_t<std::is_reference_v<T>, T, T&>;
  using pointer_type = value_type*;
  using coroutine_handle =
    std::experimental::coroutine_handle<generator_promise<T>>;

  generator_promise() noexcept
    : _value(nullptr)
    , _parent(this)
    , _child(this)
    , _exception(nullptr){};

  auto get_return_object() noexcept { return generator<T>{*this}; }

  void return_void() noexcept {}

  constexpr auto initial_suspend() const
  {
    return std::experimental::suspend_always{};
  }
  constexpr auto final_suspend() const
  {
    return std::experimental::suspend_always{};
  }

  template<typename U = T,
           std::enable_if_t<!std::is_same<typename std::remove_pointer<U>::type,
                                          generator<T>>(),
                            int> = 0>
  auto yield_value(U&& v) noexcept
  {
    _value = std::addressof(v);
    return std::experimental::suspend_always{};
  }

  template<typename U = T,
           std::enable_if_t<!std::is_rvalue_reference<U>::value, int> = 0>
  auto yield_value(generator<T>& generator) noexcept
  {
    if (generator._promise) {
      _parent->_child = generator._promise;
      generator._promise->_parent = _parent;
      generator._promise->_child = this;
      generator._promise->resume();
      if (!generator._promise->done()) {
        return awaitable<T>{generator._promise};
      }
    }
    return awaitable<T>{nullptr};
  }
  auto yield_value(generator<T>&& generator) noexcept
  {
    return yield_value(generator);
  }

  void unhandled_exception() { _exception = std::current_exception(); }

  void rethrow_if_exception()
  {
    if (_exception) { std::rethrow_exception(_exception); }
  }

  bool done() noexcept { return coroutine_handle::from_promise(*this).done(); }

  void destroy() noexcept { coroutine_handle::from_promise(*this).destroy(); }

  void resume() noexcept { coroutine_handle::from_promise(*this).resume(); }

  reference_type value() noexcept
  {
    assert(this == _parent && !this->done());
    return *(_child->_value);
  }

  template<typename U>
  std::experimental::suspend_never await_transform(U&& value) = delete;

  generator_promise* pull() noexcept
  {
    assert(this == _parent && !_child->done());
    _child->resume();

    while (_child != this && _child->done()) {
      _child = _child->_child;
      _child->resume();
    }
    return this;
  }

private:
  pointer_type _value;
  std::exception_ptr _exception;

  generator_promise* _parent;
  generator_promise* _child;
};

template<typename T>
class generator_iterator : public std::iterator<std::input_iterator_tag, T>
{

public:
  using iterator_category = std::input_iterator_tag;
  using difference_type = std::size_t;
  using value_type = typename generator_promise<T>::value_type;
  using reference_type = typename generator_promise<T>::reference_type;
  using pointer_type = typename generator_promise<T>::pointer_type;
  using coroutine_handle =
    std::experimental::coroutine_handle<generator_promise<T>>;

  generator_iterator() noexcept
    : _promise(nullptr)
  {}

  explicit generator_iterator(generator_promise<T>* promise) noexcept
    : _promise(promise)
  {}

  bool operator==(const generator_iterator& rhs) const noexcept
  {
    return _promise == rhs._promise;
  }

  bool operator!=(const generator_iterator& rhs) const noexcept
  {
    return _promise != rhs._promise;
  }

  generator_iterator* operator++()
  {
    assert(_promise && !_promise->done());
    if (_promise->pull()->done()) { _promise = nullptr; }
    return this;
  }

  void operator++(int) { operator++(); }

  reference_type operator*() const noexcept
  {
    assert(_promise);
    return static_cast<reference_type>(_promise->value());
  }

  pointer_type operator->() const noexcept
  {
    assert(_promise);
    return static_cast<reference_type>(std::addressof(_promise->value()));
  }

private:
  generator_promise<T>* _promise;
};

template<typename T>
class generator
{
public:
  using promise_type = generator_promise<T>;
  using iterator = generator_iterator<T>;
  using value_type = typename generator_promise<T>::value_type;
  using reference_type = typename generator_promise<T>::reference_type;
  using pointer_type = typename generator_promise<T>::pointer_type;

  generator() noexcept
    : _promise(nullptr)
  {}
  generator(promise_type& promise) noexcept
    : _promise(&promise)
  {}
  generator(generator&& generator) noexcept
  {
    _promise = generator._promise;
    generator._promise = nullptr;
  }
  generator(const generator& generator) = delete;

  ~generator()
  {
    if (_promise) { _promise->destroy(); }
  }

  generator& operator=(const generator& rhs) = delete;

  generator& operator=(generator&& rhs) noexcept
  {
    if (this != &rhs) {
      if (_promise) { _promise->destroy(); }
      _promise = rhs._promise;
      rhs._promise = nullptr;
    }
    return *this;
  }

  iterator begin()
  {
    if (_promise != nullptr) {
      _promise->pull();
      if (!_promise->done()) { return generator_iterator<T>(_promise); }
      _promise->rethrow_if_exception();
    }
    return generator_iterator<T>(nullptr);
  }
  iterator end() noexcept { return generator_iterator<T>(nullptr); }

  iterator begin() const { return begin(); }
  iterator end() const { return end(); }

  iterator cbegin() { return begin(); }
  iterator cend() { return end(); }

private:
  friend class generator_promise<T>;
  friend class generator_iterator<T>;

  promise_type* _promise;
};
};
#endif // GENERATOR_H
