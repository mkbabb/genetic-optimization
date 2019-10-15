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

auto
make_genes(std::vector<erate_t>& erate_data, int bucket_count)
  -> std::vector<std::tuple<int, int>>
{
    std::vector<std::tuple<int, int>> genes;
    genes.reserve(erate_data.size());

    for (auto d : erate_data) {
        std::tuple<int, int> tup;
        if (d.no_mutate) {
            tup = std::make_tuple(d.no_mutate, d.bucket);
        } else {
            tup = std::make_tuple(0, 0);
        }
        genes.push_back(tup);
    }
    return genes;
}

template<typename T>
auto
calc_pool_fitness(std::vector<erate_t>& erate_data,
                  std::map<size_t, std::map<std::string, double>>& buckets,
                  std::vector<Critter>& critters,
                  Critter& max_critter,
                  size_t max_bucket,
                  T& rng) -> Critter&

{
    for (auto& critter : critters) {
        auto fitness = 0.0;

        for (auto i : itertools::range(erate_data.size())) {
            auto j = std::get<0>(critter.genes()[i])
                       ? std::get<1>(critter.genes()[i])
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
                  double_round((bucket["total_discount"] /
                                (bucket["count"] * 100)),
                               3);
                bucket["discount_cost"] =
                  bucket["average_discount"] * bucket["total_cost"];
                fitness += bucket["discount_cost"];
            }
        }

        critter.fitness(fitness);

        max_critter =
          critter.fitness() > max_critter.fitness() ? critter : max_critter;

        for (auto& [_, bucket] : buckets) {
            for (auto& [key, value] : bucket) { value = 0; }
        }
    }
    return max_critter;
}

template<typename T>
auto
proportionate_selection(std::vector<Critter>& critters,
                        double max_fitness,
                        size_t mating_pool_count,
                        T& rng) -> std::vector<Critter>
{
    auto divisor = critters.size() * 100;
    std::map<int, int> probability_dict;
    std::vector<Critter> parents;

    auto total_fitness = 0.0;
    for (auto& critter : critters) {
        critter.fitness((critter.fitness() / max_fitness - 0.9) * 10);
        total_fitness += critter.fitness();
    }
    auto p = 0.0;
    auto prev_p = 0.0;
    for (auto [n, critter] : itertools::enumerate(critters)) {
        p = prev_p + critter.fitness() / total_fitness;
        probability_dict[n] = static_cast<int>(p * divisor);
        prev_p = p;
    }

    for (auto i : itertools::range(mating_pool_count)) {
        auto r = rng.randrange(0, divisor);
        for (auto& [key, value] : probability_dict) {
            if (r < value) {
                parents.push_back(critters[key]);
                break;
            }
        }
    }
    return parents;
}

template<typename T>
void
k_point_crossover(std::vector<Critter>& parents,
                  std::vector<Critter>& children,
                  std::vector<int>& probability_distb,
                  std::vector<int>& ixs,
                  size_t N,
                  T& rng)
{
    for (auto i : itertools::range(parents.size())) {
        auto crossover_point = rng.randrange(0, N);
        probability_distb[i] = crossover_point;
    }
    probability_distb.push_back(N);
    std::sort(begin(probability_distb), end(probability_distb));

    auto start = 0;
    for (auto end : probability_distb) {
        do {
            for (auto [n, j] : itertools::enumerate(ixs)) {
                std::get<0>(children[j].genes()[start]) =
                  std::get<0>(parents[n].genes()[start]);
                std::get<1>(children[j].genes()[start]) =
                  std::get<1>(parents[n].genes()[start]);
            }
        } while (start++ < end);
        itertools::roll(ixs);
        start = end;
    }
}

template<typename T>
void
cull_mating_pool(std::vector<erate_t>& erate_data,
                 std::vector<Critter>& critters,
                 size_t bucket_count,
                 double max_fitness,
                 size_t mutation_count,
                 size_t parent_count,
                 size_t mating_pool_count,
                 T& rng)
{
    auto N = erate_data.size();
    auto population_count = critters.size();

    auto parents =
      proportionate_selection(critters, max_fitness, mating_pool_count, rng);
    std::vector<Critter> children(parent_count * population_count, {N});

    std::vector<Critter> t_parents(begin(parents),
                                   begin(parents) + parent_count);
    std::vector<Critter> t_children(begin(children),
                                    begin(children) + parent_count);

    std::vector<int> probability_distb(parent_count, 0);
    std::vector<int> ixs(parent_count, 0);
    itertools::for_each(ixs, [](auto n, auto&& i) {
        i = n;
        return false;
    });

    auto k = 0;
    for (auto i : itertools::range(population_count)) {
        for (auto j : itertools::range(parent_count)) {
            auto r = rng.randrange(0, mating_pool_count);
            t_children[j] = children[k++];
            t_parents[j] = parents[r];
        }
        k_point_crossover(t_parents,
                          t_children,
                          probability_distb,
                          ixs,
                          N,
                          rng);
    }
    critters = std::move(children);
}

void
optimize_buckets(std::vector<erate_t>& erate_data,
                 std::string out_file,
                 size_t bucket_count,
                 size_t max_bucket,
                 size_t population_count,
                 size_t mutation_rate,
                 size_t mutation_threshold_low,
                 size_t mutation_threshold_high,
                 size_t parent_count,
                 size_t mating_pool_count,
                 size_t iterations,
                 uint64_t rng_state)
{
    std::ofstream ofs;

    auto N = erate_data.size();

    auto mutation_count =
      std::min(static_cast<size_t>((N * mutation_rate) / 100.f), N);
    mating_pool_count = std::min(mating_pool_count, N);
    max_bucket = std::min(N, max_bucket);
    auto max_fitness = 0.0;

    random_v::Random rng(rng_state, random_v::lcg_xor_rot);

    std::map<size_t, std::map<std::string, double>> buckets;
    for (auto i : itertools::range(bucket_count)) {
        buckets.emplace(i,
                        std::map<std::string, double>{{"average_discount", 0.0},
                                                      {"total_discount", 0.0},
                                                      {"total_cost", 0.0},
                                                      {"discount_cost", 0.0},
                                                      {"count", 0.0}});
    }
    std::vector<Critter> critters(population_count,
                                  {N, make_genes(erate_data, bucket_count)});
    auto& max_critter = critters[0];

    for (auto i : itertools::range(iterations)) {
        max_critter = calc_pool_fitness(erate_data,
                                        buckets,
                                        critters,
                                        max_critter,
                                        max_bucket,
                                        rng);

        if (max_fitness < max_critter.fitness()) {
            std::ofstream ofs;
            ofs.open(out_file, std::ios::trunc);
            std::string header =
              fmt::format("\niteration: {0}, discount-total: "
                          "{1:.2f}\n"
                          "max-delta: {2:.2f}, prev-max-delta: "
                          "{3:.2f}\n"
                          "iteration-delta: {4}, rng-state: {5}",
                          i,
                          max_fitness,
                          max_fitness - MAX_2019,
                          max_fitness - max_critter.fitness(),
                          i,
                          rng.state());

            max_fitness = max_critter.fitness();

            fmt::print("{0}\n", header);
            ofs << "lea-number,discount,cost,no-mutate,bucket\n";

            fmt::memory_buffer row;
            itertools::for_each(max_critter.genes(), [&](auto n, auto i) {
                fmt::format_to(row,
                               "{0},{1},{2},{3},{4}\n",
                               erate_data[n].lea_number,
                               erate_data[n].discount,
                               erate_data[n].cost,
                               std::get<0>(i),
                               std::get<1>(i));
                std::get<0>(i) = 1;
                return false;
            });

            ofs << row.data();
            ofs.close();
        }

        cull_mating_pool(erate_data,
                         critters,
                         bucket_count,
                         max_critter
                           .fitness(), // Potentially change this to max_fitness
                         mutation_count,
                         parent_count,
                         mating_pool_count,
                         rng);
    }
}

auto
process_erate_data(const std::string& in_file) -> std::vector<erate_t>
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
    while (in.read_row(lea_number, discount, cost, no_mutate, bucket)) {
        erate_data.push_back(
          erate_t{lea_number, discount, cost, no_mutate, bucket});
    }
    return erate_data;
}

int
main(int argc, char** argv)
{
    using namespace std::chrono;
    /*
    Argument parsing of argv.
     */
    cxxopts::Options options("erate", "to optimize data_t");
    options.allow_unrecognised_options()
      .add_options()("i,in_file", "Input file", cxxopts::value<std::string>())(
        "o,out_file",
        "Output file",
        cxxopts::value<std::string>())("bucket_count",
                                       "Bucket count",
                                       cxxopts::value<int>()->default_value(
                                         "4"))("max_bucket",
                                               "Max bucket count",
                                               cxxopts::value<int>()
                                                 ->default_value("150"))(
        "population_count",
        "Population count",
        cxxopts::value<int>()->default_value(
          "100"))("mutation_rate",
                  "Mutation rate",
                  cxxopts::value<int>()->default_value(
                    "1"))("mutation_threshold_low",
                          "Mutation threshold low",
                          cxxopts::value<int>()->default_value(
                            "100000"))("mutation_threshold_high",
                                       "Mutation threshold high",
                                       cxxopts::value<int>()->default_value(
                                         "1000000"))(
        "parent_count",
        "Parent count",
        cxxopts::value<int>()->default_value(
          "2"))("mating_pool_count",
                "Mating pool count",
                cxxopts::value<int>()->default_value(
                  "10"))("iterations",
                         "Iterations",
                         cxxopts::value<int>()->default_value(
                           "1000000"))("rng_state",
                                       "Random number generator state",
                                       cxxopts::value<int>()->default_value(
                                         "-1"));

    auto result = options.parse(argc, argv);

    auto erate_data = process_erate_data(result["in_file"].as<std::string>());

    auto epoch_time_us =
      duration_cast<microseconds>(system_clock::now().time_since_epoch());

    auto _rng_state = result["rng_state"].as<int>();
    auto rng_state = _rng_state > 0 ? static_cast<uint64_t>(_rng_state)
                                    : epoch_time_us.count();

    optimize_buckets(erate_data,
                     result["out_file"].as<std::string>(),
                     result["bucket_count"].as<int>(),
                     result["max_bucket"].as<int>(),
                     result["population_count"].as<int>(),
                     result["mutation_rate"].as<int>(),
                     result["mutation_threshold_low"].as<int>(),
                     result["mutation_threshold_high"].as<int>(),
                     result["parent_count"].as<int>(),
                     result["mating_pool_count"].as<int>(),
                     result["iterations"].as<int>(),
                     rng_state);
}