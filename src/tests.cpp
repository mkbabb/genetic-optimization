#define FMT_HEADER_ONLY

#include "../external/cxxopts.hpp"
#include "./erate_genetic.cpp"

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

int
main(int argc, char** argv)
{
    // std::string in_file = "./data/erate-2019-start.csv";

    // uint64_t rng_state = 1;

    // int bucket_count = 5;
    // int max_bucket = 280;
    // int population_count = 500;
    // int mutation_rate = 1;

    // int mutation_threshold_low = 1000;
    // int mutation_threshold_high = 0;

    // int parent_count = 10;
    // int mating_pool_count = 50;
    // int crossover_count = 100;
    // int iterations = 100'000;

    // auto erate_data = process_erate_data(in_file);

    // optimize_buckets(erate_data,
    //                  "",
    //                  bucket_count,
    //                  max_bucket,
    //                  population_count,
    //                  mutation_rate,
    //                  mutation_threshold_low,
    //                  mutation_threshold_high,
    //                  parent_count,
    //                  crossover_count,
    //                  mating_pool_count,
    //                  iterations,
    //                  rng_state);

    using namespace std::chrono;
    //     /*
    //     Argument parsing of argv.
    //      */
    cxxopts::Options options("erate", "to optimize data_t");
    options.allow_unrecognised_options()
      .add_options()("i,in_file",
                     "Input file",
                     cxxopts::value<std::string>())("o,out_file",
                                                    "Output file",
                                                    cxxopts::value<
                                                      std::string>())(
        "load_file",
        "Load file",
        cxxopts::value<std::string>()->default_value(
          ""))("bucket_count",
               "Bucket count",
               cxxopts::value<int>()->default_value(
                 "4"))("max_bucket",
                       "Max bucket count",
                       cxxopts::value<int>()->default_value(
                         "150"))("population_count",
                                 "Population count",
                                 cxxopts::value<int>()->default_value("100"))(
        "mutation_rate",
        "Mutation rate",
        cxxopts::value<int>()->default_value(
          "1"))("parent_count",
                "Parent count",
                cxxopts::value<int>()->default_value(
                  "2"))("crossover_count",
                        "Crossover count",
                        cxxopts::value<int>()->default_value(
                          "2"))("mating_pool_count",
                                "Mating pool count",
                                cxxopts::value<int>()->default_value(
                                  "10"))("iterations",
                                         "Iterations",
                                         cxxopts::value<int>()->default_value(
                                           "1000000"))(
        "rng_state",
        "Random number generator state",
        cxxopts::value<int>()->default_value(
          "-1"))("current_best",
                 "Current best value to optimize",
                 cxxopts::value<double>()->default_value(
                   "-1"))("nuke_threshold",
                          "nuke factor low",
                          cxxopts::value<int>()->default_value(
                            "5"))("nuke_threshold_max",
                                  "nuke factor low",
                                  cxxopts::value<int>()->default_value("5000"))(
        "nuke_mutation_percent",
        "nuke factor low",
        cxxopts::value<int>()->default_value(
          "2"))("nuke_mutation_percent_max",
                "nuke factor low",
                cxxopts::value<int>()->default_value(
                  "2"))("nuke_growth_rate",
                        "nuke threshold",
                        cxxopts::value<double>()->default_value(
                          "2"))("nuke_burnout",
                                "nuke threshold",
                                cxxopts::value<int>()->default_value("10"));

    auto result = options.parse(argc, argv);

    auto erate_data = process_erate_data(result["in_file"].as<std::string>());

    auto epoch_time_us =
      duration_cast<microseconds>(system_clock::now().time_since_epoch());

    auto _rng_state = result["rng_state"].as<int>();
    auto rng_state = _rng_state > 0 ? static_cast<uint64_t>(_rng_state)
                                    : epoch_time_us.count();

    optimize_buckets(erate_data,
                     result["out_file"].as<std::string>(),
                     result["load_file"].as<std::string>(),
                     result["bucket_count"].as<int>(),
                     result["max_bucket"].as<int>(),
                     result["population_count"].as<int>(),

                     result["mutation_rate"].as<int>(),

                     result["nuke_threshold"].as<int>(),
                     result["nuke_threshold_max"].as<int>(),
                     result["nuke_mutation_percent"].as<int>(),
                     result["nuke_mutation_percent_max"].as<int>(),
                     result["nuke_growth_rate"].as<double>(),
                     result["nuke_burnout"].as<int>(),

                     result["parent_count"].as<int>(),
                     result["crossover_count"].as<int>(),
                     result["mating_pool_count"].as<int>(),
                     result["iterations"].as<int>(),
                     rng_state,
                     result["current_best"].as<double>());
}