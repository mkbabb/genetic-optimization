#define FMT_HEADER_ONLY

#include "../external/csv.hpp"
#include "../external/cxxopts.hpp"
#include "../external/fmt/format.h"
#include "../external/itertools/src/itertools.hpp"
#include "../external/itertools/src/math.hpp"
#include "../external/random_v/src/random_v.hpp"
#include "utils.cpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <random>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <vector>

constexpr int MAX_2015 = 11'810'384;
constexpr int MAX_2019 = 15'034'520;
constexpr int MAX_2019_6 = 10'764'932;

constexpr int PROPORTIANATE_PRECISION = 100'000;

auto GAUSSIAN_SQRT2_INV = gaussian(1, sqrt2, 1, 0, sqrt2_2, true);

using namespace std::literals;

struct erate_t
{
    std::string lea_number;
    int discount, cost, no_mutate, bucket;
};

class Critter
{
  public:
    std::vector<int> _genes;
    double _fitness;
    bool _skip;

    Critter(size_t N)
      : _fitness(0)
      , _skip(false)
    {
        _genes.resize(N);
        std::fill(begin(_genes), end(_genes), 0);
    }

    Critter(size_t N, std::vector<int>&& genes)
      : _fitness(0)
      , _skip(false)
    {
        _genes.resize(N);
        this->genes(genes);
    }

    auto genes() -> std::vector<int>& { return _genes; }
    void genes(const std::vector<int>& g)
    {
        std::copy(begin(g), end(g), begin(_genes));
    }

    double fitness() { return _fitness; }
    void fitness(double f) { _fitness = f; }

    bool skip() { return _skip; }
    bool skip() const { return _skip; }
    void skip(bool s) { _skip = s; }
};

template<typename T>
auto
make_genes(std::vector<erate_t>& erate_data, int bucket_count, T& rng)
  -> std::vector<int>
{
    std::vector<int> genes(erate_data.size(), 0);

    for (auto [n, d] : itertools::enumerate(erate_data)) {
        int r = n < erate_data.size() / 2 ? 0 : rng.randrange(0, bucket_count);
    }
    return genes;
}

auto
process_erate_data(std::string in_file, int count = -1) -> std::vector<erate_t>
{
    std::vector<erate_t> erate_data;
    std::string lea_number;
    int discount, cost, no_mutate, bucket;

    io::CSVReader<5> in(in_file);
    in.read_header(io::ignore_extra_column,
                   "lea-number",
                   "discount",
                   "cost",
                   "no-mutate",
                   "bucket");

    count = count == -1 ? 0 : count;

    while (in.read_row(lea_number, discount, cost, no_mutate, bucket) &&
           (--count) != 0) {
        erate_data.push_back(
          erate_t{lea_number, discount, cost, no_mutate, bucket});
    }
    return erate_data;
}

template<typename T>
void
calc_critter_fitness(std::vector<erate_t>& erate_data,
                     Critter& critter,
                     std::map<size_t, std::map<std::string, double>>& buckets,
                     size_t max_bucket,
                     bool* randomize,
                     T& rng)
{
    auto fitness = 0.0;

    if (critter.skip() && !(*(randomize))) {
        fitness = critter.fitness();
    } else {
        for (auto [n, school] : itertools::enumerate(erate_data)) {
            auto r = !(*(randomize)) ? critter.genes()[n]
                                     : rng.randrange(0, buckets.size());

            while (buckets[r]["count"] >= max_bucket) {
                r = rng.randrange(0, buckets.size());
            }

            auto& bucket = buckets[r];

            bucket["total_cost"] += school.cost;
            bucket["total_discount"] += school.discount;
            bucket["count"]++;

            critter.genes()[n] = r;
        }

        for (auto& [_, bucket] : buckets) {
            if (bucket["count"] > 0) {
                bucket["average_discount"] =
                  bucket["total_discount"] / (bucket["count"] * 100);

                bucket["average_discount"] =
                  double_round(bucket["average_discount"], 3);

                bucket["discount_cost"] =
                  bucket["average_discount"] * bucket["total_cost"];

                fitness += bucket["discount_cost"];
            }
        }
        critter.fitness(fitness);
    }
    critter.skip(false);

    for (auto& [_, bucket] : buckets) {
        for (auto& [key, value] : bucket) {
            value = 0;
        }
    }
}

template<typename T>
void
calc_pool_fitness(std::vector<erate_t>& erate_data,
                  std::map<size_t, std::map<std::string, double>>& buckets,
                  std::vector<Critter>& critters,
                  size_t max_bucket,
                  bool* randomize,
                  T& rng)

{
    for (auto& critter : critters) {
        calc_critter_fitness(erate_data,
                             critter,
                             buckets,
                             max_bucket,
                             randomize,
                             rng);
    }
    *randomize = false;

    std::sort(begin(critters), end(critters), [](auto& c1, auto& c2) {
        return c1.fitness() > c2.fitness();
    });
}

template<typename T>
auto
proportionate_selection(std::vector<Critter>& critters,
                        double max_fitness,
                        T& rng)
{
    std::map<int, int> probability_dict;

    auto total_fitness = 0.0;
    for (auto& critter : critters) {
        auto g_fitness = pow((critter.fitness() / max_fitness), 100);

        critter.fitness(g_fitness);
        total_fitness += critter.fitness();
    }

    auto p = 0.0;
    auto prev_p = 0.0;

    for (auto [n, critter] : itertools::enumerate(critters)) {
        p = prev_p + (critter.fitness() / total_fitness);
        probability_dict[n] =
          static_cast<int>(ceil(p * PROPORTIANATE_PRECISION));
        prev_p = p;
    }

    return probability_dict;
}

template<typename T>
void
k_point_crossover(std::vector<Critter*>& parents,
                  std::vector<Critter*>& children,
                  std::vector<size_t>& crossover_distb,
                  std::vector<size_t>& parent_ixs,
                  size_t crossover_count,
                  size_t N,
                  T& rng)
{
    for (auto i : itertools::range(crossover_count - 1)) {
        auto crossover_point = rng.randrange(0, N);
        crossover_distb[i] = crossover_point;
    }

    crossover_distb[parents.size()] = N - 1;

    std::sort(begin(crossover_distb), end(crossover_distb));

    auto start = 0;
    for (auto end : crossover_distb) {
        for (auto [n, j] : itertools::enumerate(parent_ixs)) {
            auto k = start;
            while (k++ < end) {
                children[j]->genes()[k] = parents[n]->genes()[k];
            }
        }
        itertools::roll(parent_ixs);
        start = end;
    }
}

template<typename T>
void
get_critter_subset(std::vector<Critter>& critters,
                   std::vector<Critter*>& subset,
                   size_t count,
                   T& rng)
{
    for (auto i : itertools::range(count)) {
        auto r = rng.randrange(0, critters.size());
        subset[i] = &critters[r];
    }
}

template<typename T>
void
mutate_critters(std::vector<erate_t>& erate_data,
                std::vector<Critter*>& critters,
                size_t bucket_count,
                size_t mutation_count,
                T& rng)
{
    for (auto& critter : critters) {
        for (auto i : itertools::range(mutation_count)) {
            auto r = rng.randrange(0, erate_data.size());
            auto b = rng.randrange(0, bucket_count);
            critter->genes()[r] = b;
        }
    }
}

template<typename T>
void
mate(std::vector<erate_t>& erate_data,
     std::map<size_t, std::map<std::string, double>>& buckets,
     std::vector<Critter>& critters,
     std::vector<size_t>& parent_ixs,
     std::vector<size_t>& crossover_distb,
     size_t bucket_count,
     double max_fitness,
     size_t max_bucket,
     size_t population_count,
     size_t mutation_count,
     size_t parent_count,
     size_t crossover_count,
     size_t mating_pool_count,
     T& rng)
{
    auto probability_dict = proportionate_selection(critters, max_fitness, rng);

    std::vector<Critter> children(parent_count * population_count,
                                  {erate_data.size()});

    std::vector<Critter*> t_parents(parent_count);
    std::vector<Critter*> t_children(parent_count);

    auto k = 0;
    for (auto i : itertools::range(population_count)) {
        for (auto j : itertools::range(parent_count)) {
            t_children[j] = &children[k];

            // Actual proportionate selection.
            auto r = rng.randrange(0, PROPORTIANATE_PRECISION);
            for (auto [key, value] : probability_dict) {
                if (r < value) {
                    t_parents[j] = &critters[key];
                    break;
                }
            }
            k++;
        }
        k_point_crossover(t_parents,
                          t_children,
                          crossover_distb,
                          parent_ixs,
                          crossover_count,
                          erate_data.size(),
                          rng);

        mutate_critters(erate_data, t_children, bucket_count, 1, rng);
    }
    bool randomize = false;

    calc_pool_fitness(erate_data,
                      buckets,
                      children,
                      max_bucket,
                      &randomize,
                      rng);

    itertools::for_each(critters, [&](auto n, auto v) {
        critters[n] = std::move(children[n]);
    });
}

void
max_critter_to_csv(std::vector<erate_t>& erate_data,
                   Critter& max_critter,
                   double prev_max_fitness,
                   std::string& out_file,
                   std::ofstream& ofs,
                   size_t i,
                   size_t max_gap,
                   double current_best)
{
    ofs.open(out_file, std::ios::trunc);

    auto max_delta = max_critter.fitness() - prev_max_fitness;
    auto total_savings = max_critter.fitness() - current_best;

    std::cout << fmt::format("i: {0:n}; max-gap: {1:n}; max-fitness: {2:.3f}; "
                             "max-delta: {3:.3f}; total-savings: {4:.3f}\n",
                             i,
                             max_gap,
                             max_critter.fitness(),
                             max_delta,
                             total_savings);

    ofs << "lea-number,discount,cost,no-mutate,bucket\n";

    fmt::memory_buffer row;

    for (auto [n, gene] : itertools::enumerate(max_critter.genes())) {
        fmt::format_to(row,
                       "{0},{1},{2},{3},{4}\n",
                       erate_data[n].lea_number,
                       erate_data[n].discount,
                       erate_data[n].cost,
                       0,
                       gene);
    }
    ofs << row.data();
    ofs.close();
}

void
optimize_buckets(std::vector<erate_t>& erate_data,
                 std::string out_file,
                 std::string load_file,
                 size_t bucket_count,
                 size_t max_bucket,
                 size_t population_count,
                 size_t mutation_rate,
                 size_t mutation_threshold_low,
                 size_t mutation_threshold_high,
                 size_t parent_count,
                 size_t crossover_count,
                 size_t mating_pool_count,
                 size_t iterations,
                 uint64_t rng_state,
                 double current_best)
{
    std::ofstream ofs;

    bool randomize = true;

    std::vector<Critter> critters(population_count, {erate_data.size()});
    std::vector<Critter*> critter_subset(mating_pool_count);

    random_v::Random rng(rng_state, random_v::lcg_xor_rot);

    if (load_file != "") {
        randomize = false;
        auto t_erate_data = process_erate_data(load_file);
        auto genes = make_genes(t_erate_data, bucket_count, rng);
        for (auto& critter : critters) {
            critter.genes(genes);
        }
    }

    auto mutation_count =
      std::min(static_cast<size_t>((erate_data.size() * mutation_rate) / 100.0),
               erate_data.size());

    mating_pool_count = std::min(mating_pool_count, population_count);
    parent_count = std::min(parent_count, mating_pool_count);
    max_bucket = std::min(erate_data.size(), max_bucket);
    crossover_count = std::max(parent_count + 1, crossover_count);

    auto prev_max_fitness = 0.0;

    std::map<size_t, std::map<std::string, double>> buckets;
    for (auto i : itertools::range(bucket_count)) {
        buckets.emplace(i,
                        std::map<std::string, double>{{"average_discount", 0.0},
                                                      {"total_discount", 0.0},
                                                      {"total_cost", 0.0},
                                                      {"discount_cost", 0.0},
                                                      {"count", 0.0}});
    }

    std::vector<size_t> parent_ixs(parent_count, 0);
    itertools::for_each(parent_ixs, [&](auto n, auto v) { parent_ixs[n] = n; });

    std::vector<size_t> crossover_distb(crossover_count, 0);

    auto max_gap = 0;

    for (auto i : itertools::range(iterations)) {
        if (max_gap > mutation_threshold_low) {
            get_critter_subset(critters,
                               critter_subset,
                               critter_subset.size(),
                               rng);
            mutate_critters(erate_data,
                            critter_subset,
                            bucket_count,
                            mutation_count,
                            rng);
            max_gap = 0;
        } else {
            max_gap++;
        }

        calc_pool_fitness(erate_data,
                          buckets,
                          critters,
                          max_bucket,
                          &randomize,
                          rng);

        auto& max_critter = critters[0];

        if (prev_max_fitness < max_critter.fitness()) {
            max_critter_to_csv(erate_data,
                               max_critter,
                               prev_max_fitness,
                               out_file,
                               ofs,
                               i,
                               max_gap,
                               current_best);

            prev_max_fitness = max_critter.fitness();
            max_gap = 0;
        }

        mate(erate_data,
             buckets,
             critters,
             parent_ixs,
             crossover_distb,
             bucket_count,
             max_critter.fitness(), // Potentially change this to max_fitness
             max_bucket,
             population_count,
             mutation_count,
             parent_count,
             crossover_count,
             mating_pool_count,
             rng);
    }
}