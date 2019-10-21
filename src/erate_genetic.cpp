#define FMT_HEADER_ONLY

#include "../external/csv.hpp"
#include "../external/cxxopts.hpp"
#include "../external/fmt/format.h"
#include "../external/itertools/src/itertools.hpp"
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
using namespace std::literals;

struct erate_t
{
    std::string lea_number;
    int discount, cost, no_mutate, bucket;
};

class Critter
{
  public:
    std::vector<std::tuple<int, int>> _genes;
    double _fitness;
    bool _skip;

    Critter(size_t N)
      : _fitness(0)
      , _skip(false)
    {
        _genes.resize(N);
        std::fill(begin(_genes), end(_genes), std::forward_as_tuple(0, 0));
    }

    Critter(size_t N, std::vector<std::tuple<int, int>>&& g)
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
    void genes(const std::vector<std::tuple<int, int>>& g)
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

template<typename T>
auto
make_genes(std::vector<erate_t>& erate_data, int bucket_count, T& rng)
  -> std::vector<std::tuple<int, int>>
{
    std::vector<std::tuple<int, int>> genes;
    genes.reserve(erate_data.size());
    int n = 0;
    for (auto d : erate_data) {
        if (n < erate_data.size() / 2) {
            auto tup = std::make_tuple(d.no_mutate, d.bucket);
            genes.push_back(tup);
        } else {
            auto r = rng.randrange(0, bucket_count);
            auto tup = std::make_tuple(0, r);
            genes.push_back(tup);
        }
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
calc_pool_fitness(std::vector<erate_t>& erate_data,
                  std::map<size_t, std::map<std::string, double>>& buckets,
                  std::vector<Critter>& critters,
                  Critter* max_critter,
                  size_t max_bucket,
                  bool* randomize,
                  T& rng) -> Critter*

{
    for (auto& critter : critters) {
        auto fitness = 0.0;

        for (auto i : itertools::range(erate_data.size())) {
            auto j = !(*(randomize)) ? std::get<1>(critter.genes()[i])
                                     : rng.randrange(0, buckets.size());

            while (buckets[j]["count"] >= max_bucket) {
                j = rng.randrange(0, buckets.size());
            }

            auto& bucket = buckets[j];

            bucket["total_cost"] += erate_data[i].cost;
            bucket["total_discount"] += erate_data[i].discount;
            bucket["count"]++;

            std::get<1>(critter.genes()[i]) = j;
        }

        for (auto& [_, bucket] : buckets) {
            if (bucket["count"] > 0) {
                bucket["average_discount"] =
                  double_round(bucket["total_discount"] /
                                 (bucket["count"] * 100),
                               3); // maybe change this.

                bucket["discount_cost"] =
                  bucket["average_discount"] * bucket["total_cost"];

                fitness += bucket["discount_cost"];
            }
        }

        critter.fitness(fitness);

        max_critter =
          critter.fitness() > max_critter->fitness() ? &critter : max_critter;

        for (auto& [_, bucket] : buckets) {
            for (auto& [key, value] : bucket) { value = 0; }
        }
    }
    *randomize = false;
    return max_critter;
}

template<typename T>
auto
proportionate_selection(std::vector<Critter>& critters,
                        double max_fitness,
                        size_t mating_pool_count,
                        T& rng) -> std::vector<Critter*>
{
    std::map<int, double> probability_dict;
    std::vector<Critter*> parents(mating_pool_count);

    auto total_fitness = 0.0;
    for (auto& critter : critters) {
        critter.fitness(((critter.fitness() / max_fitness - 0.999) * 1000));
        total_fitness += critter.fitness();
    }

    auto p = 0.0;
    auto prev_p = 0.0;
    int n = 0;
    for (auto& critter : critters) {
        p = prev_p + (critter.fitness() / total_fitness);
        probability_dict[n] = p;
        prev_p = p;
        n++;
    }

    for (auto i : itertools::range(mating_pool_count)) {
        auto r = rng.unit();
        for (auto [key, value] : probability_dict) {
            if (r < value) {
                parents[i] = &critters[key];
                break;
            }
        }
    }
    return parents;
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
        int n = 0;
        for (auto j : parent_ixs) {
            auto k = start;
            while (k++ < end) {
                std::get<0>(children[j]->genes()[k]) =
                  std::get<0>(parents[n]->genes()[k]);
                std::get<1>(children[j]->genes()[k]) =
                  std::get<1>(parents[n]->genes()[k]);
            }
            n++;
        }
        itertools::roll(parent_ixs);
        start = end;
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
            std::get<1>(critter->genes()[r]) = rng.randrange(0, bucket_count);
        }
    }
}

template<typename T>
void
cull_mating_pool(std::vector<Critter>& critters,
                 std::vector<Critter>& children,
                 T& rng)
{
    int n = 0;
    for (auto& critter : critters) {
        auto r = rng.randrange(0, children.size());
        critters[n] = children[r];
        n++;
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

    std::sort(begin(critters), end(critters), [](auto c1, auto c2) {
        return c1.fitness() > c2.fitness();
    });

    // auto parents =
    //   proportionate_selection(critters, max_fitness, mating_pool_count, rng);

    std::vector<Critter> parents(begin(critters),
                                 begin(critters) + mating_pool_count);

    std::vector<Critter> children(parent_count * population_count, {N});

    std::vector<Critter*> t_parents(parent_count);
    std::vector<Critter*> t_children(parent_count);

    auto k = 0;
    for (auto i : itertools::range(population_count)) {
        for (auto j : itertools::range(parent_count)) {
            auto r = rng.randrange(0, mating_pool_count);
            t_children[j] = &children[k];
            t_parents[j] = &parents[r];
            k++;
        }
        k_point_crossover(t_parents,
                          t_children,
                          crossover_distb,
                          parent_ixs,
                          crossover_count,
                          N,
                          rng);

        mutate_critters(t_children, bucket_count, mutation_count, N, rng);
    }
    // bool randomize = false;
    // calc_pool_fitness(erate_data,
    //                   buckets,
    //                   children,
    //                   &children[0],
    //                   max_bucket,
    //                   &randomize,
    //                   rng);

    // std::sort(begin(children), end(children), [](auto c1, auto c2) {
    //     return c1.fitness() > c2.fitness();
    // });
    cull_mating_pool(critters, children, rng);
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
    random_v::Random rng(rng_state, random_v::lcg_xor_rot);

    if (load_file != "") {
        randomize = false;
        auto t_erate_data = process_erate_data(load_file);
        auto genes = make_genes(t_erate_data, bucket_count, rng);
        for (auto& critter : critters) { critter.genes(genes); }
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
    for (auto i : itertools::range(parent_count)) { parent_ixs[i] = i; }
    std::vector<size_t> crossover_distb(crossover_count, 0);

    int max_gap = 0;

    for (auto i : itertools::range(iterations)) {
        if (max_gap > mutation_threshold_low) {
            randomize = true;
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

            max_fitness = max_critter->fitness();
            std::cout << fmt::
                format("max-gap: {}, i: {}, max-fitness: {}, max-delta: {}\n",
                       max_gap,
                       i,
                       max_fitness,
                       max_fitness - MAX_2019);
            max_gap = 0;

            ofs << "lea-number,discount,cost,no-mutate,bucket\n";

            fmt::memory_buffer row;
            int n = 0;
            for (auto& i : max_critter->genes()) {
                fmt::format_to(row,
                               "{0},{1},{2},{3},{4}\n",
                               erate_data[n].lea_number,
                               erate_data[n].discount,
                               erate_data[n].cost,
                               std::get<0>(i),
                               std::get<1>(i));
                std::get<0>(i) = 1;
                n++;
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

// int
// main(int argc, char** argv)
// {
//     using namespace std::chrono;
//     /*
//     Argument parsing of argv.
//      */
//     cxxopts::Options options("erate", "to optimize data_t");
//     options.allow_unrecognised_options()
//       .add_options()("i,in_file", "Input file",
//       cxxopts::value<std::string>())(
//         "o,out_file",
//         "Output file",
//         cxxopts::value<std::string>())("bucket_count",
//                                        "Bucket count",
//                                        cxxopts::value<int>()->default_value(
//                                          "4"))("max_bucket",
//                                                "Max bucket count",
//                                                cxxopts::value<int>()
//                                                  ->default_value("150"))(
//         "population_count",
//         "Population count",
//         cxxopts::value<int>()->default_value(
//           "100"))("mutation_rate",
//                   "Mutation rate",
//                   cxxopts::value<int>()->default_value(
//                     "1"))("mutation_threshold_low",
//                           "Mutation threshold low",
//                           cxxopts::value<int>()->default_value(
//                             "100000"))("mutation_threshold_high",
//                                        "Mutation threshold high",
//                                        cxxopts::value<int>()->default_value(
//                                          "1000000"))(
//         "parent_count",
//         "Parent count",
//         cxxopts::value<int>()->default_value(
//           "2"))("mating_pool_count",
//                 "Mating pool count",
//                 cxxopts::value<int>()->default_value(
//                   "10"))("iterations",
//                          "Iterations",
//                          cxxopts::value<int>()->default_value(
//                            "1000000"))("rng_state",
//                                        "Random number generator state",
//                                        cxxopts::value<int>()->default_value(
//                                          "-1"));

//     auto result = options.parse(argc, argv);

//     auto erate_data =
//     process_erate_data(result["in_file"].as<std::string>());

//     auto epoch_time_us =
//       duration_cast<microseconds>(system_clock::now().time_since_epoch());

//     auto _rng_state = result["rng_state"].as<int>();
//     auto rng_state = _rng_state > 0 ? static_cast<uint64_t>(_rng_state)
//                                     : epoch_time_us.count();

//     optimize_buckets(erate_data,
//                      result["out_file"].as<std::string>(),
//                      result["bucket_count"].as<int>(),
//                      result["max_bucket"].as<int>(),
//                      result["population_count"].as<int>(),
//                      result["mutation_rate"].as<int>(),
//                      result["mutation_threshold_low"].as<int>(),
//                      result["mutation_threshold_high"].as<int>(),
//                      result["parent_count"].as<int>(),
//                      result["mating_pool_count"].as<int>(),
//                      result["iterations"].as<int>(),
//                      rng_state);
// }