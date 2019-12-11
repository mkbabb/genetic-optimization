union double_bits
{
    double value;
    uint64_t ivalue;

    constexpr double_bits(double d)
      : value(d){};
};

std::tuple<uint, int, uint64_t>
frexp0(double d)
{
    uint sign = 0;
    int exponent;

    double dmantissa = frexp(d, &exponent);

    if (dmantissa < 0) {
        dmantissa *= -1;
        sign = 1;
    }

    uint64_t mantissa = dmantissa * two53;

    return {sign, exponent, mantissa};
}

constexpr std::tuple<uint, int, uint64_t>
frexp1(double d)
{
    uint64_t ivalue = 0;

    double_bits db(d);

    uint sign = db.ivalue >> 63;
    int exponent = (db.ivalue << 1) >> 53;
    uint64_t mantissa = (db.ivalue << 12) >> 12;

    if (exponent == 0) {
        exponent++;
    } else {
        mantissa |= (1L << 52);
    }
    exponent -= 1022;

    return {sign, exponent, mantissa};
}

std::tuple<uint, int, uint64_t>
frexp3(double d)
{
    uint64_t ivalue = *((uint64_t*) (&d));

    double mantissa = 1.0;

    uint sign = ivalue >> 63;
    int exponent = (int) ((ivalue >> 52) & 0x7ffL);
    uint64_t imantissa = ivalue & 0xfffffffffffffL;

    if (exponent == 0) {
        exponent++;
    } else {
        imantissa |= (1L << 52);
    }

    // bias the exponent - actually biased by 1023.
    // we are treating the mantissa as m.0 instead of 0.m
    //  so subtract another 52.
    exponent -= 1075;
    mantissa = imantissa;
    uint64_t mant = imantissa;

    // normalize
    while (mantissa >= 1.0) {
        imantissa >>= 1;
        mantissa /= 2.;
        exponent++;
    }

    if (sign) {
        mantissa *= -1;
    }

    return {sign, exponent, mant};
}

void
frexp_tests()
{
    random_v::Random rng(0ULL, random_v::lcg_xor_rot, 0);
    int a = -100'000;
    int b = 100'000;

    for (double i : itertools::range(100'000)) {
        double d = static_cast<double>(rng.randrange(a, b)) / 10000.0;

        auto tup0 = frexp0(d);
        auto tup1 = frexp1(d);
        auto tup2 = cfrexp(d);
        auto tup3 = frexp3(d);

        assert(tup0 == tup1);
        assert(tup0 == tup2);
        assert(tup0 == tup3);

        // std::cout << itertools::to_string(tup0) << std::endl;
        // std::cout << itertools::to_string(tup1) << std::endl;
        // std::cout << itertools::to_string(tup2) << std::endl;
        // std::cout << itertools::to_string(tup3) << std::endl;
    }
}