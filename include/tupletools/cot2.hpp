#include <exception>
#include <experimental/coroutine>
#include <iostream>
#include <type_traits>
#include <utility>

template<typename T>
class generator;

template<typename T>
class generator_promise
{
public:
  using value_type = std::remove_reference_t<T>;
  using reference_type = std::conditional_t<std::is_reference_v<T>, T, T&>;
  using pointer_type = value_type*;
  using coroutine_handle =
    std::experimental::coroutine_handle<generator_promise<T>>;

  generator_promise() = default;
  generator<T> get_return_object() noexcept
  {
    return generator<T>{coroutine_handle::from_promise(*this)};
  }

  constexpr auto initial_suspend() const -> std::experimental::suspend_always
  {
    return {};
  }
  constexpr auto final_suspend() const -> std::experimental::suspend_always
  {
    return {};
  }

  template<typename U = T,
           std::enable_if_t<!std::is_rvalue_reference<U>::value, int> = 0>
  auto yield_value(T& v) -> std::experimental::suspend_always
  {
    this->current_value = std::addressof(v);
    return {};
  }
  auto yield_value(T&& v) -> std::experimental::suspend_always
  {
    this->current_value = std::addressof(v);
    return {};
  }

  auto yield_value(generator<T>&& generator) noexcept
  {
    return yield_value(generator);
  }

  auto yield_value(generator<T>& generator) noexcept
  {
    struct awaitable
    {

      awaitable(generator_promise<T>* childPromise)
        : m_childPromise(childPromise)
      {}

      bool await_ready() noexcept { return this->m_childPromise == nullptr; }

      void await_suspend(coroutine_handle) noexcept {}

      void await_resume()
      {
        if (this->m_childPromise != nullptr) {
          this->m_childPromise->rethrow_if_exception();
        }
      }

    private:
      generator_promise<T>* m_childPromise;
    };

    std::cout << "him2" << std::endl;

    if (generator.current_coroutine != nullptr) {
      generator.current_coroutine.resume();
      if (!generator.current_coroutine.done()) {
        return awaitable{&generator.current_coroutine.promise()};
      }
    }
    return awaitable{nullptr};
  }

  void unhandled_exception() { current_exception = std::current_exception(); }

  void rethrow_if_exception()
  {
    if (current_exception) { std::rethrow_exception(this->current_exception); }
  }

  void return_void() {}

  reference_type value() const noexcept { return *(current_value); }

  // Don't allow any use of 'co_await' inside the generator coroutine.
  template<typename U>
  std::experimental::suspend_never await_transform(U&& value) = delete;

private:
  pointer_type current_value;
  std::exception_ptr current_exception;
};

class generator_sentinel
{};

template<typename T>
class generator_iterator
{
  using coroutine_handle =
    std::experimental::coroutine_handle<generator_promise<T>>;

public:
  using iterator_category = std::input_iterator_tag;
  using difference_type = std::size_t;
  using value_type = typename generator_promise<T>::value_type;
  using reference_type = typename generator_promise<T>::reference_type;
  using pointer_type = typename generator_promise<T>::pointer_type;

  generator_iterator() noexcept { this->current_coroutine(nullptr); }

  explicit generator_iterator(coroutine_handle coroutine) noexcept
    : current_coroutine(coroutine)
  {}

  bool operator==(const generator_iterator& other) const noexcept
  {
    return this->current_coroutine == other.current_coroutine;
  }

  bool operator==(generator_sentinel) const noexcept
  {
    return !this->current_coroutine || this->current_coroutine.done();
  }

  bool operator!=(const generator_iterator& other) const noexcept
  {
    return !(*this == other);
  }

  bool operator!=(generator_sentinel other) const noexcept
  {
    return !operator==(other);
  }

  generator_iterator& operator++()
  {
    this->current_coroutine.resume();
    if (this->current_coroutine.done()) {
      this->current_coroutine.promise().rethrow_if_exception();
    }
    return *this;
  }

  void operator++(int) { (void) this->operator++(); }

  reference_type operator*() const noexcept
  {
    return current_coroutine.promise().value();
  }

  pointer_type operator->() const noexcept { return std::addressof(*this); }

private:
  coroutine_handle current_coroutine;
};

template<typename T>
class [[nodiscard]] generator
{
public:
  using promise_type = generator_promise<T>;
  using iterator = generator_iterator<T>;

  generator() noexcept
    : current_coroutine(nullptr)
  {}

  generator(generator && other) noexcept
    : current_coroutine(other.current_coroutine)
  {
    other.current_coroutine = nullptr;
  }

  generator(const generator& other) = delete;

  ~generator()
  {
    if (current_coroutine) { current_coroutine.destroy(); }
  }

  generator& operator=(generator other) noexcept
  {
    std::swap(current_coroutine, other.current_coroutine);
    return *this;
  }

  iterator begin()
  {
    if (current_coroutine) {
      current_coroutine.resume();
      if (current_coroutine.done()) {
        current_coroutine.promise().rethrow_if_exception();
      }
    }
    return iterator{current_coroutine};
  }

  generator_sentinel end() noexcept { return generator_sentinel{}; }

private:
  friend class generator_promise<T>;
  std::experimental::coroutine_handle<promise_type> current_coroutine;

  explicit generator(decltype(current_coroutine) coroutine) noexcept
    : current_coroutine(coroutine)
  {}
};
