#define FMT_HEADER_ONLY

#include "csv.hpp"
#include "fmt/format.h"
#include "random_t/lcg.cpp"
#include "random_t/xor_shift.cpp"
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

uint64_t STATE = 0xDEADBEEF;
uint64_t STATE2 = 0xDEADBEEFED;
uint64_t STATE3 = 0xFEEDBEEF;

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
  int discount, cost;
};

class Critter
{
public:
  std::vector<std::tuple<int, int>> genes;
  double fitness;
  bool skip;

  Critter(int N)
    : fitness(0)
    , skip(false)
  {
    genes.resize(N);
    std::fill(begin(genes), end(genes), std::forward_as_tuple(0, 0));
  }

  friend std::ostream& operator<<(std::ostream& os, Critter const& data)
  {
    fmt::memory_buffer c;
    fmt::format_to(
      c, "{{\nfitness: {0},\nskip: {1},\ngenes:\n\t[", data.fitness, data.skip);
    for (auto& [g1, g2] : data.genes) {
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
    int j = std::get<0>(critter.genes[i])
              ? std::get<1>(critter.genes[i])
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
    std::get<1>(critter.genes[i]) = j;
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
    critter.fitness = fitness;
  }
}

template<typename Func>
void
op_critter(Critter& c1, Critter& c2, Func func)
{
  for (int i = 0; i < c1.genes.size(); i++) {
    func(c1, c2, i);
  }
}

auto
parent_distb(int N) -> std::vector<int>
{
  std::vector<int> t(N);

  if (N == 1) {
    t[0] = 1;
  } else {
    int first = 80;
    int second = 20;
    int third = 10;

    t[0] = first - 10 * (N - 2);
    t[1] = second;
    for (int i = 0; i < N - 3; i++) {
      t[i] = third;
    }
  }
  return t;
}

void
mate_t(Critter& child,
       std::vector<Critter>& parents,
       std::vector<int>& pdistb,
       int N)
{
  std::sort(begin(parents), end(parents), [](auto c1, auto c2) {
    return c1.fitness > c2.fitness;
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
    std::get<0>(child.genes[i]) = std::get<0>(parents[parent].genes[i]);
    std::get<1>(child.genes[i]) = std::get<1>(parents[parent].genes[i]);
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
    return c1.fitness > c2.fitness;
  });

  std::vector<Critter> children(begin(critters), begin(critters) + top_pool);
  std::vector<Critter> parents(begin(critters), begin(critters) + parent_count);
  for (auto& child : children) {
    child.skip = true;
  }

  for (int i = 0; i < M - top_pool; i++) {
    Critter child(N);
    child.skip = false;

    for (int j = 0; j < parent_count; j++) {
      int r = randrange(0, M - 1);
      parents[j] = critters[r];
    }

    // std::copy(
    //   begin(parents[0].genes), end(parents[0].genes), begin(child.genes));

    mate_t(child, parents, pdistb, N);

    for (int j = 0; j < mut_count; j++) {
      int k = randrange(0, N);
      std::get<0>(child.genes[k]) ^= 1;
    }
    children.push_back(child);
  }
  critters = std::move(children);
}

auto
optimize_buckets(std::vector<erate_t> data,
                 int bucket_count,
                 int pop_count = 10,
                 int mut_rate = 1,
                 int parent_count = 2,
                 int iterations = 1000,
                 int mut_threshold = 1000,
                 int max_bucket = 0,
                 bool save_critter = false) -> void
{
  std::ofstream myfile;

  int N = data.size();
  int top_pool = pop_count / 10;
  max_bucket = std::max(max_bucket, N / bucket_count);
  int mut_count = static_cast<float>(N * mut_rate) / 100;
  double max_fitness = 0;

  std::vector<int> pdistb = parent_distb(parent_count);
  std::map<int, std::map<std::string, double>> buckets;
  std::vector<int> full_buckets(bucket_count, 0);

  for (int i = 0; i < bucket_count; i++) {
    buckets[i] = std::map<std::string, double>{ { "average_discount", 0 },
                                                { "total_discount", 0 },
                                                { "total_cost", 0 },
                                                { "discount_cost", 0 },
                                                { "count", 0 } };
  }
  std::vector<Critter> critters;
  for (int i = 0; i < pop_count; i++) {
    critters.emplace_back(N);
  }

  int l = 0;
  int mut_gap = 0;
  int t_mut_count = mut_count;

  for (int i = 0; i < iterations; i++) {
    for (auto& critter : critters) {
      if (!critter.skip) {
        calc_erate_stats(
          data, bucket_count, max_bucket, N, critter, buckets, full_buckets);

        if (critter.fitness > max_fitness) {
          myfile.open("./erate-max-2015.csv", std::ios::trunc);

          std::string header =
            fmt::format("--- iteration: {0}, discount-total: {1:.2f} ---\n"
                        "--- max-delta: {2:.2f}, p-max-delta: {3:.2f}, "
                        "iteration-delta: {4} ---",
                        i,
                        critter.fitness,
                        critter.fitness - MAX_2019,
                        critter.fitness - max_fitness,
                        i - mut_gap);

          max_fitness = critter.fitness;
          l = i;
          mut_gap = i;

          myfile << header + "\n";
          std::cout << header << std::endl;

          myfile << "lea-number,discount,cost,bucket\n";
          fmt::memory_buffer row;
          for (int k = 0; k < N; k++) {
            fmt::format_to(row,
                           "{0},{1},{2},{3}\n",
                           data[k].lea_number,
                           data[k].discount,
                           data[k].cost,
                           std::get<1>(critter.genes[k]));
            std::get<0>(critter.genes[k]) = 1;
          }
          myfile << row.data();
          if (save_critter) {
            myfile << "\n--- critters of iteration " << i << " ---\n";
            myfile << "{";
            for (int k = 0; k < pop_count; k++) {
              myfile << "\ncritter_" << k << ": " << critters[k] << ",";
            }
            myfile << "}";
          }
          myfile.close();
        }
      } else {
        critter.skip = true;
      }
    }
    if ((i - l) > mut_threshold) {
      l = i;
      t_mut_count = static_cast<float>(N * bucket_count * mut_count) / 100;
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
main()
{
  io::CSVReader<3> in("./erate-data-2019.csv");
  in.read_header(io::ignore_extra_column, "lea-number", "discount", "cost");
  std::vector<erate_t> erate_data;

  std::string lea_number;
  int discount, cost;

  int i = 0;
  int N = 281;

  while (in.read_row(lea_number, discount, cost) && i++ < N) {
    erate_data.push_back(erate_t{ lea_number, discount, cost });
  }

  int bucket_count = 4;
  int pop_count = 100;
  int max_bucket = 150;
  int mut_rate = 1;
  int parent_count = 9;
  int iterations = 10'000'000;
  int mut_threshold = 1'000;

  optimize_buckets(erate_data,
                   bucket_count,
                   pop_count,
                   mut_rate,
                   parent_count,
                   iterations,
                   mut_threshold,
                   max_bucket);

  //   int n = 32;
  //   int lgn = log2(n);

  //   auto t = de_bruijn(2, lgn);
  //   int db = std::stoi(t, 0, 2);

  //   std::cout << t << std::endl;

  //   auto tab = tabn(n, db);

  //   std::cout << ranges::join(tab, ", ") << std::endl;
  //   return 0;
}