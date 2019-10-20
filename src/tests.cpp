#define FMT_HEADER_ONLY

#include "../external/csv.hpp"
#include "../external/cxxopts.hpp"
#include "../external/fmt/format.h"
#include "../external/itertools/src/itertools.hpp"
#include "../external/random_v/src/random_v.hpp"
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
main()
{
    std::string in_file = "./data/erate-2019-start.csv";

    uint64_t rng_state = 1;

    int bucket_count = 4;
    int max_bucket = 200;
    int population_count = 500;
    int mutation_rate = 1;
    
    int mutation_threshold_low = 0;
    int mutation_threshold_high = 0;
   
    int parent_count = 2;
    int mating_pool_count = 100;
    int iterations = 10'000;

    auto erate_data = process_erate_data(in_file);
    
    optimize_buckets(erate_data,
                     "",
                     bucket_count,
                     max_bucket,
                     population_count,
                     mutation_rate,
                     mutation_threshold_low,
                     mutation_threshold_high,
                     parent_count,
                     mating_pool_count,
                     iterations,
                     rng_state);
}