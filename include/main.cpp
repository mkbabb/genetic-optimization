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

uint32_t
rand2m(uint8_t m, uint64_t* state)
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
randrange(uint32_t a, uint32_t b)
{
  auto rng = [&]() { return lcg_xor_rot(&::STATE2); };
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

  Critter(int N)
    : fitness(0)
  {
    for (int i = 0; i < N; i++) {
      genes.push_back({ 0, 0 });
    }
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
    int j = std::get<0>(critter.genes[i]) ? std::get<1>(critter.genes[i])
                                          : randrange(0, bucket_count);

    if (full_buckets[j]) {
      while (full_buckets[j]) {
        j = randrange(0, bucket_count);
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
        double_round(b["total_discount"] / (100 * b["count"]), 2);
      b["discount_cost"] =
        double_round(b["average_discount"] * b["total_cost"]);
    }
    fitness += b["discount_cost"];
    full_buckets[i] = 0;

    for (auto& [key, value] : b) {
      b[key] = 0;
    }
  }
  critter.fitness = fitness;
}

template<typename Func>
void
op_critter(Critter c1, Critter c2, Func func)
{
  for (int i = 0; i < c1.genes.size(); i++) {
    func(c1, c2, i);
  }
}

void
mate_critters(std::vector<erate_t>& data,
              int max_fitness,
              int mut_count,
              int parent_count,
              int N,
              std::vector<Critter>& critters)
{
  int M = critters.size();

  double ratio = 0;
  for (auto& critter : critters) {
    if (critter.fitness > 0) {
      critter.fitness = (critter.fitness / max_fitness - 0.9) * 10;
      ratio += critter.fitness > 0 ? 1 / critter.fitness : 0;
    }
  }

  std::sort(begin(critters), end(critters), [](auto c1, auto c2) {
    return c1.fitness > c2.fitness;
  });

  //   int L = M * ratio;
  //   std::map<int, int> weights;
  //   int k = 0;
  //   for (int i = 0; i < M; i++) {
  //     int count = L * critters[i].fitness;
  //     for (int j = 0; j < count; j++) {
  //       weights[k] = i;
  //       k += 1;
  //     }
  //   }

  std::vector<Critter> children(begin(critters),
                                begin(critters) + parent_count);

  for (int i = 0; i < M - parent_count; i++) {
    Critter child(N);
    for (int j = 0; j < parent_count; j++) {
      int r = randrange(0, M - 1);
      auto parent = critters[r];
      if (j == 0) {
        std::copy(begin(parent.genes), end(parent.genes), begin(child.genes));
      } else {
        op_critter(child, parent, [](auto& c1, auto& c2, int i) {
          std::get<0>(c1.genes[i]) &= std::get<0>(c2.genes[i]);
        });
      }
    }
    for (int j = 0; j < mut_count; j++) {
      int k = randrange(0, N);
      std::get<0>(child.genes[k]) = randrange(0, 1);
    }
    children.push_back(child);
  }
  critters = std::move(children);
}

auto
optimize_buckets(std::vector<erate_t> data,
                 int bucket_count,
                 int max_bucket,
                 int pop_count = 10,
                 int mut_rate = 1,
                 int parent_count = 2,
                 int iterations = 1000) -> void
{

  int N = data.size();
  max_bucket = std::max(max_bucket, N / bucket_count);
  int mut_count = static_cast<float>(N * mut_rate) / 100;
  double max_fitness = 0;

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

  for (int i = 0; i < iterations; i++) {
    for (auto& critter : critters) {
      calc_erate_stats(
        data, bucket_count, max_bucket, N, critter, buckets, full_buckets);

      if (critter.fitness > max_fitness) {
        std::ofstream myfile;
        myfile.open("./erate-max-2015.csv",
                    std::ofstream::out | std::ofstream::trunc);

        std::cout << critter.fitness << " " << i << std::endl;
        max_fitness = critter.fitness;

        myfile << "lea-number,discount,cost,bucket\n";
        fmt::memory_buffer row;

        for (int k = 0; k < N; k++) {
          fmt::format_to(row,
                         "{0},{1},{2},{3}\n",
                         data[k].lea_number,
                         data[k].discount,
                         data[k].cost,
                         std::get<1>(critter.genes[k]));
          myfile << row.data();
          std::get<0>(critter.genes[k]) = 1;
        }
      };
    }
    mate_critters(data, max_fitness, mut_count, parent_count, N, critters);
  }
}

const int MAX_2015 = 11810384;

int
main()
{
  io::CSVReader<3> in("./erate-data-2015.csv");
  in.read_header(io::ignore_extra_column, "lea-number", "discount", "cost");
  std::vector<erate_t> erate_data;

  std::string lea_number;
  int discount, cost;

  int i = 0;
  int N = 221;

  while (in.read_row(lea_number, discount, cost) && i++ < N) {
    erate_data.push_back(erate_t{ lea_number, discount, cost });
  }
  std::cout << erate_data.size() << std::endl;

  optimize_buckets(erate_data, 3, 100, 100, 1, 2, 1'000'000);

  //   int n = 32;
  //   int lgn = log2(n);

  //   auto t = de_bruijn(2, lgn);
  //   int db = std::stoi(t, 0, 2);

  //   std::cout << t << std::endl;

  //   auto tab = tabn(n, db);

  //   std::cout << ranges::join(tab, ", ") << std::endl;
  //   return 0;
}