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
auto
calc_critter_fitness((std::vector<erate_t>& erate_data,
                    Critter& critter,
                     std::map<size_t, std::map<std::string, double>>& buckets,
                     size_t max_bucket,
                     bool* randomize,
                     T& rng) -> void
{
    auto fitness = 0.0;

    if (critter.skip() && !(*(randomize))) {
        fitness = critter.fitness();
    } else {
        for (auto& [n, school] : itertools::enumerate(erate_data)) {
            auto r = !(*(randomize)) ? critter.genes()[i]
                                     : rng.randrange(0, buckets.size());

            while (buckets[r]["count"] >= max_bucket) {
                r = rng.randrange(0, buckets.size());
            }

            auto& bucket = buckets[r];

            bucket["total_cost"] += school.cost;
            bucket["total_discount"] += school.discount;
            bucket["count"]++;

            critter.genes()[i] = r;
        }

        for (auto& [_, bucket] : buckets) {
            if (bucket["count"] > 0) {
                bucket["average_discount"] =
                  bucket["total_discount"] / (bucket["count"] * 100);

                bucket["average_discount"] =
                  double_round(bucket["average_discount"], 2);

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
auto
calc_pool_fitness(std::vector<erate_t>& erate_data,
                  std::map<size_t, std::map<std::string, double>>& buckets,
                  std::vector<Critter>& critters,
                  Critter* max_critter,
                  size_t max_bucket,
                  bool* randomize,
                  T& rng) -> Critter*

{
    for (auto& critter : critters) {
        calc_critter_fitness(critter, buckets, max_bucket, randomize, rng);

        max_critter =
          critter.fitness() > max_critter->fitness() ? &critter : max_critter;
    }
    *randomize = false;
    return max_critter;
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
        auto g_fitness =
          powc(GAUSSIAN_SQRT2_INV, (critter.fitness() / max_fitness));

        critter.fitness(g_fitness);
        total_fitness += critter.fitness();
    }

    auto p = 0.0;
    auto prev_p = 0.0;
    for (auto [n, critter] : itertools::enumerate(critters)) {
        p = prev_p + (critter.fitness() / total_fitness);
        probability_dict[n] = static_cast<int>(ceil(p * 100'000));
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
mutate_critters(std::vector<Critter*>& critters,
                size_t bucket_count,
                size_t mutation_count,
                size_t N,
                T& rng)
{
    for (auto& critter : critters) {
        for (auto i : itertools::range(mutation_count)) {
            auto r = rng.randrange(0, N);
            auto b = rng.randrange(0, bucket_count);
            critter->genes()[r] = b;
        }
    }
}

void
cull_mating_pool(std::vector<Critter>& critters, std::vector<Critter>& children)
{
    for (auto [n, critter] : itertools::enumerate(critters)) {
        critters[n] = children[n];
        critters[n].skip(true);
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
     size_t mutation_count,
     size_t parent_count,
     size_t crossover_count,
     size_t mating_pool_count,
     T& rng)
{
    auto N = erate_data.size();
    auto population_count = critters.size();

    std::sort(begin(critters), end(critters), [](auto& c1, auto& c2) {
        return c1.fitness() > c2.fitness();
    });

    auto probability_dict = proportionate_selection(critters, max_fitness, rng);

    std::vector<Critter> children(parent_count * population_count, {N});

    std::vector<Critter*> t_parents(parent_count);
    std::vector<Critter*> t_children(parent_count);

    auto k = 0;
    for (auto i : itertools::range(population_count)) {
        for (auto j : itertools::range(parent_count)) {
            t_children[j] = &children[k];

            // Actual proportionate selection.
            auto r = rng.randrange(0, 100'000);
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
                          N,
                          rng);

        mutate_critters(t_children, bucket_count, 1, N, rng);
    }
    bool randomize = false;
    calc_pool_fitness(erate_data,
                      buckets,
                      children,
                      &children[0],
                      max_bucket,
                      &randomize,
                      rng);
    std::sort(begin(children), end(children), [](auto& c1, auto& c2) {
        return c1.fitness() > c2.fitness();
    });
    cull_mating_pool(critters, children);
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
                 uint64_t rng_state)
{
    std::ofstream ofs;
    auto N = erate_data.size();
    bool randomize = true;

    std::vector<Critter> critters(population_count, {N});
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
      std::min(static_cast<size_t>((N * mutation_rate) / 100.f), N);

    mating_pool_count = std::min(mating_pool_count, N);
    parent_count = std::min(parent_count, mating_pool_count);
    max_bucket = std::min(N, max_bucket);
    crossover_count = std::max(parent_count + 1, crossover_count);

    auto max_fitness = 0.0;

    std::map<size_t, std::map<std::string, double>> buckets;
    for (auto i : itertools::range(bucket_count)) {
        buckets.emplace(i,
                        std::map<std::string, double>{{"average_discount", 0.0},
                                                      {"total_discount", 0.0},
                                                      {"total_cost", 0.0},
                                                      {"discount_cost", 0.0},
                                                      {"count", 0.0}});
    }
    auto max_critter = &critters[0];

    std::vector<size_t> parent_ixs(parent_count, 0);
    for (auto i : itertools::range(parent_count)) {
        parent_ixs[i] = i;
    }
    std::vector<size_t> crossover_distb(crossover_count, 0);

    int max_gap = 0;

    for (auto i : itertools::range(iterations)) {
        if (max_gap > mutation_threshold_low) {
            get_critter_subset(critters,
                               critter_subset,
                               critter_subset.size(),
                               rng);
            mutate_critters(critter_subset,
                            bucket_count,
                            mutation_count,
                            N,
                            rng);
            max_gap = 0;
        }
        max_critter = calc_pool_fitness(erate_data,
                                        buckets,
                                        critters,
                                        max_critter,
                                        max_bucket,
                                        &randomize,
                                        rng);

        if (max_fitness < max_critter->fitness()) {
            ofs.open(out_file, std::ios::trunc);

            std::cout
              << fmt::format("i: {0:n}; max-gap: {1:n}; max-fitness: {2:n}; "
                             "max-delta: {3:n}; total-savings: {4:n}\n",
                             i,
                             max_gap,
                             static_cast<int>(max_critter->fitness()),
                             static_cast<int>(max_critter->fitness() -
                                              max_fitness),
                             static_cast<int>(max_critter->fitness() -
                                              MAX_2019));
            max_gap = 0;
            max_fitness = max_critter->fitness();

            ofs << "lea-number,discount,cost,no-mutate,bucket\n";

            fmt::memory_buffer row;

            for (auto [n, i] : itertools::enumerate(max_critter->genes())) {
                fmt::format_to(row,
                               "{0},{1},{2},{3},{4}\n",
                               erate_data[n].lea_number,
                               erate_data[n].discount,
                               erate_data[n].cost,
                               0,
                               i);
            }
            ofs << row.data();
            ofs.close();
        }
        max_gap++;

        mate(erate_data,
             buckets,
             critters,
             parent_ixs,
             crossover_distb,
             bucket_count,
             max_critter->fitness(), // Potentially change this to max_fitness
             max_bucket,
             mutation_count,
             parent_count,
             crossover_count,
             mating_pool_count,
             rng);
    }
}