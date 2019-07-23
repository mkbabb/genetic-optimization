#define FMT_HEADER_ONLY

#include "csv.hpp"
#include "cxxopts.hpp"
#include "fmt/format.h"
#include "random_t/lcg.cpp"
#include "random_t/xor_shift.cpp"
#include "tupletools.hpp"
#include "utils.cpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <vector>

uint64_t STATE1 = 0xDEADBEEF;
uint64_t STATE2 = 0xDEADBEEFED;
uint64_t STATE3 = 0xFEEDBEEF;
uint64_t STATE4 = 0xABBADEAF;

constexpr int MAX_2015 = 11'810'384;
constexpr int MAX_2019 = 15'034'520;

uint32_t
rand2m(uint8_t m, uint64_t* state = &::STATE3)
{
  return lcg_xor_rot(state) >> (31 - m);
};

template<typename F>
uint32_t
bounded_rand(F& rng, uint32_t range)
{
  uint32_t t = (-range) % range;
  while (true) {
    uint32_t r = rng();
    if (r >= t)
      return r % range;
  }
}

uint32_t
randrange(uint32_t a, uint32_t b, uint64_t* state = &::STATE2)
{
  auto rng = [&]() { return lcg_xor_rot(state); };
  uint32_t range = b - a;
  return bounded_rand(rng, range) + a;
}

struct erate_t
{
  std::string lea_number;
  int discount, cost, do_mutate, bucket;
};

class Critter
{
public:
  std::vector<std::tuple<int, int>> _genes;
  double _fitness;
  bool _skip;

  Critter(int N)
    : _fitness(0)
    , _skip(false)
  {
    _genes.resize(N);
    std::fill(begin(_genes), end(_genes), std::forward_as_tuple(0, 0));
  }

  Critter(int N, std::vector<std::tuple<int, int>>& g)
    : _fitness(0)
    , _skip(false)
  {
    _genes.resize(N);
    this->genes(g);
  }

  auto genes() -> std::vector<std::tuple<int, int>>& { return _genes; }
  auto genes() const -> const std::vector<std::tuple<int, int>>
  {
    return _genes;
  }
  void genes(std::vector<std::tuple<int, int>> const& g)
  {
    std::copy(begin(g), end(g), begin(_genes));
  }

  double fitness() { return _fitness; }
  double fitness() const { return _fitness; }
  void fitness(double f) { _fitness = f; }

  bool skip() { return _skip; }
  bool skip() const { return _skip; }
  void skip(bool s) { _skip = s; }

  friend std::ostream& operator<<(std::ostream& os, Critter const& data)
  {
    fmt::memory_buffer c;
    fmt::format_to(c,
                   "{{\nfitness: {0},\nskip: {1},\ngenes:\n\t[",
                   data.fitness(),
                   data.skip());
    for (auto& [g1, g2] : data.genes()) {
      fmt::format_to(c, "\n\t\t[{0}, {1}]", g1, g2);
    }
    fmt::format_to(c, "\n\t]\n}}");
    os << c.data();
    return os;
  }
};

auto
calc_erate_stats(std::vector<erate_t>& data,
                 int bucket_count,
                 int max_bucket,
                 int N,
                 Critter& critter,
                 std::map<int, std::map<std::string, double>>& buckets,
                 std::vector<int>& full_buckets) -> void
{
  int blog = ilog2(bucket_count) - 1;

  for (int i = 0; i < N; i++) {
    int j = std::get<0>(critter.genes()[i])
              ? std::get<1>(critter.genes()[i])
              : blog != -1 ? rand2m(blog) : randrange(0, bucket_count);

    if (full_buckets[j]) {
      while (full_buckets[j]) {
        j = blog != -1 ? rand2m(blog) : randrange(0, bucket_count);
      }
    }

    auto& b = buckets[j];
    b["total_cost"] += data[i].cost;
    b["total_discount"] += data[i].discount;
    b["count"] += 1;

    if (b["count"] > max_bucket) {
      full_buckets[j] = 1;
    }
    std::get<1>(critter.genes()[i]) = j;
  }

  double fitness = 0;
  for (int i = 0; i < bucket_count; i++) {
    auto& b = buckets[i];
    if (b["count"] > 0) {
      b["average_discount"] =
        double_round((b["total_discount"] / (b["count"] * 100)), 2);
      b["discount_cost"] = b["average_discount"] * b["total_cost"];

      fitness += b["discount_cost"];
      full_buckets[i] = 0;

      for (auto& [key, value] : b) {
        b[key] = 0;
      }
    }
    critter.fitness(fitness);
  }
}

auto
parent_distb(int N) -> std::vector<int>
{

  int slice = ceil(100.0 / N);
  if (N * slice > 100) {
    std::vector<int> t(N, slice);
    t[N - 1] = (100 - (N * (slice - 1)));
    return t;
  } else {
    std::vector<int> t(N, slice);
    return t;
  }
}

void
mate_t(Critter& child,
       std::vector<Critter>& parents,
       std::vector<int>& pdistb,
       int N)
{
  std::sort(begin(parents), end(parents), [](auto c1, auto c2) {
    return c1.fitness() > c2.fitness();
  });
  int parent = 0;

  for (int i = 0; i < N; i++) {
    int r = randrange(0, 100);
    int start = 0;

    for (int j = 0; j < parents.size(); j++) {
      if (start < r && r < (start + pdistb[j])) {
        parent = j;
        break;
      } else {
        start += pdistb[j];
      }
    }
    std::get<0>(child.genes()[i]) = std::get<0>(parents[parent].genes()[i]);
    std::get<1>(child.genes()[i]) = std::get<1>(parents[parent].genes()[i]);
  }
}

void
mate_critters(std::vector<erate_t>& data,
              int max_fitness,
              int mut_count,
              int parent_count,
              int N,
              int top_pool,
              std::vector<int>& pdistb,
              std::vector<Critter>& critters)
{
  int M = critters.size();

  std::sort(begin(critters), end(critters), [](auto c1, auto c2) {
    return c1.fitness() > c2.fitness();
  });

  std::vector<Critter> children(begin(critters), begin(critters) + top_pool);
  std::vector<Critter> parents(begin(critters), begin(critters) + parent_count);

  for (int i = 0; i < M - top_pool; i++) {
    Critter child(N);

    for (int j = 0; j < parent_count; j++) {
      int r = randrange(0, M - 1);
      parents[j] = critters[r];
    }

    mate_t(child, parents, pdistb, N);

    for (int j = 0; j < mut_count; j++) {
      int k = randrange(0, N);
      std::get<0>(child.genes()[k]) = randrange(0, 1, &::STATE4);
    }
    children.push_back(child);
  }
  critters = std::move(children);
}

auto
make_genes(std::vector<erate_t>& data, int bucket_count)
  -> std::vector<std::tuple<int, int>>
{
  std::vector<std::tuple<int, int>> genes;
  genes.reserve(data.size());

  int blog = ilog2(bucket_count) - 1;

  for (auto [n, d] : tupletools::enumerate(data)) {
    std::tuple<int, int> tup;
    if (d.do_mutate) {
      tup = std::make_tuple(d.do_mutate, d.bucket);
    } else {
      tup = std::make_tuple(
        0, blog != -1 ? rand2m(blog) : randrange(0, bucket_count));
    }
    genes.push_back(tup);
  }
  return genes;
}

void
optimize_buckets(std::vector<erate_t> data,
                 std::string out_file,
                 int bucket_count,
                 int pop_count,
                 int mut_rate,
                 int parent_count,
                 int iterations,
                 int mut_threshold_low,
                 int mut_threshold_high,
                 int max_bucket)
{
  std::ofstream ofs;

  int N = data.size();
  int top_pool = pop_count / 10;
  max_bucket = std::max(max_bucket, N / bucket_count);
  int mut_count = static_cast<float>(N * mut_rate) / 100;
  double max_fitness = 0;

  std::vector<int> pdistb = { 70, 20, 10 };
  std::map<int, std::map<std::string, double>> buckets;
  std::vector<int> full_buckets(bucket_count, 0);

  for (int i = 0; i < bucket_count; i++) {
    buckets[i] = std::map<std::string, double>{ { "average_discount", 0 },
                                                { "total_discount", 0 },
                                                { "total_cost", 0 },
                                                { "discount_cost", 0 },
                                                { "count", 0 } };
  }

  std::vector<std::tuple<int, int>> genes = make_genes(data, bucket_count);
  std::vector<Critter> critters(pop_count, { N, genes });

  int l = 0;
  int mut_gap = 0;
  int t_mut_count = mut_count;
  for (int i = 0; i < iterations; i++) {
    for (auto& critter : critters) {
      calc_erate_stats(
        data, bucket_count, max_bucket, N, critter, buckets, full_buckets);

      if (critter.fitness() > max_fitness) {
        ofs.open(out_file, std::ios::trunc);

        std::string header =
          fmt::format("--- iteration: {0}, discount-total: {1:.2f} ---\n"
                      "--- max-delta: {2:.2f}, prev-max-delta: {3:.2f}, "
                      "iteration-delta: {4} ---",
                      i,
                      critter.fitness(),
                      critter.fitness() - MAX_2019,
                      critter.fitness() - max_fitness,
                      i - mut_gap);

        max_fitness = critter.fitness();
        l = i;
        mut_gap = i;

        ofs << header + "\n";
        std::cout << header << std::endl;
        ofs << "lea-number,discount,cost,bucket\n";

        fmt::memory_buffer row;
        for (int k = 0; k < N; k++) {
          fmt::format_to(row,
                         "{0},{1},{2},{3}\n",
                         data[k].lea_number,
                         data[k].discount,
                         data[k].cost,
                         std::get<0>(critter.genes()[k]),
                         std::get<1>(critter.genes()[k]));
          std::get<0>(critter.genes()[k]) = 1;
        }
        ofs << row.data();
        ofs.close();
      }
    }
    mate_critters(data,
                  max_fitness,
                  t_mut_count,
                  parent_count,
                  N,
                  top_pool,
                  pdistb,
                  critters);
    t_mut_count = mut_count;
  }
}

int
main(int argc, char** argv)
{
  cxxopts::Options options("erate", "to optimize data_t");
  options.allow_unrecognised_options().add_options()(
    "i,in_file", "Input file", cxxopts::value<std::string>())(
    "o,out_file", "Output file", cxxopts::value<std::string>())(
    "bucket_count", "Bucket count", cxxopts::value<int>()->default_value("4"))(
    "max_bucket",
    "Max bucket count",
    cxxopts::value<int>()->default_value("100"))(
    "pop_count",
    "Population count",
    cxxopts::value<int>()->default_value("100"))(
    "mut_rate", "Mutation rate", cxxopts::value<int>()->default_value("1"))(
    "mut_threshold_low",
    "Mutation threshold low",
    cxxopts::value<int>()->default_value("100000"))(
    "mut_threshold_high",
    "Mutation threshold high",
    cxxopts::value<int>()->default_value("1000000"))(
    "parent_count", "Parent count", cxxopts::value<int>()->default_value("2"))(
    "iterations",
    "Iterations",
    cxxopts::value<int>()->default_value("1000000"));

  auto result = options.parse(argc, argv);

  std::string in_file = result["in_file"].as<std::string>();
  std::string out_file = result["out_file"].as<std::string>();

  int bucket_count = result["bucket_count"].as<int>();
  int max_bucket = result["max_bucket"].as<int>();
  int pop_count = result["pop_count"].as<int>();
  int mut_rate = result["mut_rate"].as<int>();
  int mut_threshold_low = result["mut_threshold_low"].as<int>();
  int mut_threshold_high = result["mut_threshold_low"].as<int>();
  int parent_count = result["parent_count"].as<int>();
  int iterations = result["iterations"].as<int>();

  std::vector<erate_t> erate_data;
  std::string lea_number;
  int discount, cost, do_mutate, bucket;

  io::CSVReader<5> in(in_file);
  in.read_header(io::ignore_extra_column,
                 "lea-number",
                 "discount",
                 "cost",
                 "do-mutate",
                 "bucket");

  while (in.read_row(lea_number, discount, cost, do_mutate, bucket)) {
    erate_data.push_back(
      erate_t{ lea_number, discount, cost, do_mutate, bucket });
  }

  optimize_buckets(erate_data,
                   out_file,
                   bucket_count,
                   pop_count,
                   mut_rate,
                   parent_count,
                   iterations,
                   mut_threshold_low,
                   mut_threshold_high,
                   max_bucket);
}