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

int
main()
{
    uint64_t rng_state = 1;
    random_v::Random rng(rng_state, random_v::lcg_xor_rot);
    std::string out_file = "test_rand.csv";
    std::ofstream ofs;

    ofs.open(out_file, std::ios::trunc);

    ofs << "x,y\n";

    uint64_t min = 0;
    uint64_t max = 100000;

    for (auto i : itertools::range(10000)) {
        std::string row = fmt::format("{}, {}\n",
                                      rng.randrange(min, max),
                                      rng.randrange(min, max));
        ofs << row;
    };
    ofs.close();
}