#define FMT_HEADER_ONLY

#include "../../fmt/format.h"
#include "../../random_v/src/random_v.hpp"
#include "../src/generator.hpp"
#include "../src/itertools.hpp"

#include <chrono>
#include <deque>
#include <experimental/coroutine>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <numeric>
#include <string>
#include <vector>

constexpr double ln2 = 0.693147180559945309417232121458176568075500134360;

constexpr uint64_t POW_2[64] = {9223372036854775808ULL,
                                4611686018427387904ULL,
                                2305843009213693952,
                                1152921504606846976,
                                576460752303423488,
                                288230376151711744,
                                144115188075855872,
                                72057594037927936,
                                36028797018963968,
                                18014398509481984,
                                9007199254740992,
                                4503599627370496,
                                2251799813685248,
                                1125899906842624,
                                562949953421312,
                                281474976710656,
                                140737488355328,
                                70368744177664,
                                35184372088832,
                                17592186044416,
                                8796093022208,
                                4398046511104,
                                2199023255552,
                                1099511627776,
                                549755813888,
                                274877906944,
                                137438953472,
                                68719476736,
                                34359738368,
                                17179869184,
                                8589934592,
                                4294967296,
                                2147483648,
                                1073741824,
                                536870912,
                                268435456,
                                134217728,
                                67108864,
                                33554432,
                                16777216,
                                8388608,
                                4194304,
                                2097152,
                                1048576,
                                524288,
                                262144,
                                131072,
                                65536,
                                32768,
                                16384,
                                8192,
                                4096,
                                2048,
                                1024,
                                512,
                                256,
                                128,
                                64,
                                32,
                                16,
                                8,
                                4,
                                2,
                                1};

constexpr double POW_2_M2[64] =
  {2.0000000000000000000000000000000000000000000000000000,
   1.4142135623730950488016887242096980785696718753769481,
   1.1892071150027210667174999705604759152929720924638174,
   1.0905077326652576592070106557607079789927027185400671,
   1.0442737824274138403219664787399290087846031296627133,
   1.0218971486541166782344801347832994397821404024486081,
   1.0108892860517004600204097905618605243881376678100500,
   1.0054299011128028213513839559347998147001084469362532,
   1.0027112750502024854307455884503620404730044238327597,
   1.0013547198921082058808815267840949473485306596662412,
   1.0006771306930663566781727848746471948378219842487377,
   1.0003385080526823129533054818562164040355585206751763,
   1.0001692397053022310836407580640735025136671793421835,
   1.0000846162726943132026333307835912254522374288790575,
   1.0000423072413958193392519088640529282937840441991927,
   1.0000211533969648080942498073331996562980602425520876,
   1.0000105766425497202348486284205645317969471154797808,
   1.0000052883072917631113668674642409094434692025707413,
   1.0000026441501501165475027533852261741027395384320666,
   1.0000013220742011181771202435702836050274107775676828,
   1.0000006610368820742088289260502489013734510130561973,
   1.0000003305183864159025349770924440260858530234881598,
   1.0000001652591795526530542805355470566035368192921809,
   1.0000000826295863625022559211583769929936630821530130,
   1.0000000413147923277950954161609490328414942815985912,
   1.0000000206573959505335439795206440245817777369706170,
   1.0000000103286979219257716085634456026722759399565052,
   1.0000000051643489476276357778503882281979962653238039,
   1.0000000025821744704800053909258459980727063456279533,
   1.0000000012910872344065495720391466585058062965007660,
   1.0000000006455436169949115052981368300278162892441468,
   1.0000000003227718084453649324855227384365830749678693,
   1.0000000001613859042096597612039766311019850327446120,
   1.0000000000806929521015742043425548411503818875240791,
   1.0000000000403464760499731831064518907410942175922039,
   1.0000000000201732380247831117870236677577797737198673,
   1.0000000000100866190123406859519617775814176498420247,
   1.0000000000050433095061576254905934388545557146943583,
   1.0000000000025216547530756333739498649605164483836141,
   1.0000000000012608273765370218441382198657274805228419,
   1.0000000000006304136882683122113599319045010054155322,
   1.0000000000003152068441341064280026714608185628802044,
   1.0000000000001576034220670407945820121095086394687000,
   1.0000000000000788017110335172924361751497738271146855,
   1.0000000000000394008555167578700043798486723738865748,
   1.0000000000000197004277583787409487629927863749611140,
   1.0000000000000098502138791893219610247635062123519534,
   1.0000000000000049251069395946488521731985314221271933,
   1.0000000000000024625534697973213940018034602975180717,
   1.0000000000000012312767348986599389797027787963059885,
   1.0000000000000006156383674493297799845516515601563991,
   1.0000000000000003078191837246648426159508913205936341,
   1.0000000000000001539095918623324094638942120454274986,
   1.0000000000000000769547959311662017709267976189966476,
   1.0000000000000000384773979655831001452083217085690768,
   1.0000000000000000192386989827915498875403915790522302,
   1.0000000000000000096193494913957748975042534707180385,
   1.0000000000000000048096747456978874371856411556570002,
   1.0000000000000000024048373728489437157011991829029953,
   1.0000000000000000012024186864244718571276942427201215,
   1.0000000000000000006012093432122359283831207841772167,
   1.0000000000000000003006046716061179641463788077928973,
   1.0000000000000000001503023358030589820618940078225209,
   1.0000000000000000000751511679015294910281231548927785};

// void
// zip_tests()
// {
//     {
//         std::vector<int> iv1 = {101, 102, 103, 104};
//         std::vector<int> iv2 = {9, 8, 7, 6, 5, 4, 3, 2};
//         std::vector<int> iv3 = {123, 1234, 12345};

//         for (auto [i, j, k] : itertools::zip(iv1, iv2, iv3)) {
//         };
//     }
//     {
//         std::vector<std::string> sv1 = {"\nThis",
//                                         "but",
//                                         "different",
//                                         "containers,"};
//         std::vector<std::string> sv2 = {"is", "can", "types", "too?"};
//         std::vector<std::string> sv3 = {"cool,",
//                                         "we iterate through",
//                                         "of",
//                                         "...\n"};

//         for (auto [i, j, k] : itertools::zip(sv1, sv2, sv3)) {
//         };
//     }
//     {

//         std::list<std::string> sl1 = {"Yes, we can!",
//                                       "Some more numbers:",
//                                       "More numbers!"};
//         std::vector<std::string> sv1 = {"Different types, too!",
//                                         "Ints and doubles.",
//                                         ""};
//         std::list<int> iv1 = {1, 2, 3};
//         std::vector<double> dv1{3.141592653589793238, 1.6181, 2.71828};

//         for (auto [i, j, k, l] : itertools::zip(sl1, sv1, iv1, dv1)) {
//         }
//     }
//     {
//         std::map<int, int> id1 =
//           {{0, 10}, {1, 11}, {2, 12}, {3, 13}, {4, 14}, {5, 15}};
//         std::list<std::string> sv1 =
//           {"1", "mijn", "worten", "2", "helm", "dearth"};
//         std::vector<double> dv1 = {1.2, 3.4, 5.6, 6.7, 7.8, 8.9, 9.0};

//         for (auto [i, j, k, l] :
//              itertools::zip(id1, sv1, dv1, itertools::range(7))) {
//             auto [key, value] = i;
//             fmt::print("{}: {}, {}, {}, {}\n", key, value, j, k, l);
//         }
//     }
//     {
//         std::vector<int> iv1(10, 10);
//         std::vector<int> iv2(10, 10);

//         auto tup = std::make_tuple(iv1, iv2);

//         for (auto [i, j] : itertools::zip(tup)) {
//         }
//     }
// }

// void
// tupletools_tests()
// {
//     {
//         auto tup1 = std::make_tuple(1, 2, 3, 4);
//         assert(tupletools::to_string(tup1) == "(1, 2, 3, 4)");
//     }
//     {
//         auto tup1 = tupletools::make_tuple_of<20>(true);
//         assert(tupletools::tuple_size<decltype(tup1)>::value == 20);
//     }
//     {
//         std::vector<std::tuple<int, int>> tv1 = {{1, 2},
//                                                  {3, 4},
//                                                  {5, 6},
//                                                  {7, 8}};
//         for (auto v : itertools::enumerate(tv1)) {
//             auto [n, i, j] = tupletools::flatten(v);
//             fmt::print("{} {} {}\n", n, i, j);
//         }
//     }
// }

// void
// itertools_tests()
// {
//     {
//         std::vector<std::string> sv1 = {"h", "e", "l", "l", "o"};
//         std::vector<int> iv1 = {1, 2, 3, 4, 5, 6, 7, 8};

//         assert(itertools::join(sv1, "") == "hello");
//         assert(itertools::join(sv1, ",") == "h,e,l,l,o");
//         assert(itertools::join(iv1, ", ") == "1, 2, 3, 4, 5, 6, 7, 8");
//     }
//     {
//         std::vector<int> iv1 = {1, 2, 3, 4, 5, 6, 7, 8};
//         itertools::roll(itertools::roll(iv1));
//         assert(iv1 == (std::vector<int>{7, 8, 1, 2, 3, 4, 5, 6}));
//     }
// }

// void
// any_tests()
// {
//     {
//         // Any tests with initializer list
//         auto tup1 = std::make_tuple(1, 2, 3, 4, 5, 6);
//         auto tup2 = std::make_tuple(33, 44, 77, 4, 5, 99);

//         auto tup_bool =
//           tupletools::where([](auto&& x, auto&& y) { return x == y; },
//                             tup1,
//                             tup2);
//         bool bool1 = tupletools::any_of(tup_bool);
//         bool bool2 =
//           tupletools::any_where([](auto&& x, auto&& y) { return x == y; },
//                                 tup1,
//                                 tup2);

//         assert(bool1 == true);
//         assert(bool1 = bool2);
//     }
//     {
//         // Any tests with tuple of booleans.
//         auto tup_bool1 = std::make_tuple(true, true, true, true, false);
//         auto tup_bool2 = std::make_tuple(false, false, false, false, false);

//         bool bool1 = tupletools::any_of(tup_bool1);
//         bool bool2 = tupletools::any_of(tup_bool2);

//         assert(bool1 == true);
//         assert(bool2 == false);
//     }
//     {
//         // All tests
//         auto tup1 = std::make_tuple(1, 2, 3, 4, 5, 6);
//         auto tup2 = std::make_tuple(1, 2, 3, 4, 5, 6);

//         auto tup_bool =
//           tupletools::where([](auto&& x, auto&& y) { return x == y; },
//                             tup1,
//                             tup2);
//         bool bool1 = tupletools::all_of(tup_bool);
//         bool bool2 =
//           tupletools::all_where([](auto&& x, auto&& y) { return x == y; },
//                                 tup1,
//                                 tup2);

//         assert(bool1 == true);
//         assert(bool1 = bool2);
//     }
//     {
//         // All tests with tuple of booleans.
//         auto tup_bool1 = std::make_tuple(true, true, true, true, false);
//         auto tup_bool2 = std::make_tuple(true, true, true, true, true);

//         bool bool1 = tupletools::all_of(tup_bool1);
//         bool bool2 = tupletools::all_of(tup_bool2);

//         assert(bool1 == false);
//         assert(bool2 == true);
//     }
//     {
//         // Disjunction tests
//         auto tup1 = std::make_tuple(1, 2, 3, 4);
//         auto tup2 = std::make_tuple(1, 2, 7, 4);

//         auto ilist =
//           tupletools::where([](auto&& x, auto&& y) { return x == y; },
//                             tup1,
//                             tup2);

//         bool bool1 = tupletools::disjunction_of(ilist);
//         assert(bool1 == false);
//     }
//     {
//         // Disjunction tests with tuple of booleans.
//         auto tup_bool1 = std::make_tuple(true, true, true, false, false);
//         auto tup_bool2 = std::make_tuple(true, true, false, true, true);

//         bool bool1 = tupletools::disjunction_of(tup_bool1);
//         bool bool2 = tupletools::disjunction_of(tup_bool2);

//         assert(bool1 == true);
//         assert(bool2 == false);
//     }
// }

// void
// enumerate_tests()
// {
//     auto tup = std::make_tuple(1, 2);
//     {
//         std::vector<int> v1(100000, 0);
//         int j = 0;
//         int k = 0;
//         for (auto [n, i] : itertools::enumerate(v1)) {
//             j++;
//             k = n;
//             std::get<0>(tup);
//         }
//         assert((j - 1) == k);
//     }
//     {
//         std::vector<int> v1(1000000, 0);
//         int j = 0;
//         int k = 0;

//         for (auto [n, i] : itertools::enumerate(v1)) {
//             j++;
//             k = n;
//             std::get<0>(tup);
//         }
//         assert((j - 1) == k);
//     }
// }

// void
// rvalue_zip_tests()
// {
//     {
//         std::vector<int> v1(10, 10);

//         for (auto [i, j] : itertools::zip(v1, std::vector<int>{1, 2, 3, 4}))
//         {
//             tupletools::const_downcast(i) = 99;
//         }
//         for (auto [n, i] : itertools::enumerate(v1)) {
//             if (n < 4) {
//                 assert(i == 99);
//             }
//         }
//         itertools::for_each(v1, [](auto&& n, auto&& v) {
//             tupletools::const_downcast(v) = 77;
//             return false;
//         });
//         for (auto [n, i] : itertools::enumerate(v1)) {
//             assert(i == 77);
//         }
//     }
// }

// void
// range_tests()
// {
//     {
//         int stop = -999'999;
//         int j = stop;
//         auto _range = itertools::range(stop);
//         for (auto i : _range) {
//             assert(i == j);
//             j++;
//         }
//         assert(j == 0);
//     }
//     {
//         int stop = 1'999'999;
//         int j = 0;
//         auto _range = itertools::range(stop);
//         for (auto i : _range) {
//             assert(i == j);
//             j++;
//         }
//         assert(j == stop);
//     }

//     {
//         int stop = -999'999;
//         int j = stop;
//         auto _range = itertools::range(stop, 0);
//         for (auto i : _range) {
//             assert(i == j);
//             j++;
//         }
//         assert(j == 0);
//     }
// }

// itertools::generator<int>
// inc(int n)
// {
//     for (int i = 0; i < n; i++) {
//         co_yield i;
//     };
// };

// itertools::generator<int>
// rec(int n)
// {
//     co_yield n;
//     if (n > 0) {
//         co_yield rec(n - 1);
//     }
//     co_return;
// };

// void
// rec2(int n)
// {
//     std::vector<int> t = {1, 2, 3, 4, 5, 6, 7, 8};
//     int to = 0;
//     for (auto i : itertools::range(1000)) {
//         to++;
//     }
//     for (auto i : itertools::enumerate(t)) {
//         to++;
//     }
//     if (n < 1'000) {
//         rec2(n + 1);
//     }
// };

// void
// generator_tests()
// {
//     {
//         int n = 7'000;
//         auto gen = rec(n);
//         for (auto i : gen) {
//             assert((n--) == i);
//         }
//     }
//     {
//         int n = 500'000;
//         auto gen = inc(n);
//         n = 0;
//         for (auto i : gen) {
//             assert((n++) == i);
//         }
//     }
//     {
//         rec2(0);
//     }
// }

// void
// reduction_tests()
// {
//     {
//         std::vector<std::tuple<int, int>> iter = {{0, 1},
//                                                   {1, 2},
//                                                   {3, 4},
//                                                   {5, 6},
//                                                   {7, 8}};

//         int sm = itertools::reduce<int>(iter, 0, [](auto n, auto v, auto i) {
//             return std::get<0>(v) + i;
//         });

//         assert(sm == 16);

//         int ml = itertools::reduce<int>(iter, 1, [](auto n, auto v, auto i) {
//             return std::get<1>(v) * i;
//         });

//         assert(ml == 384);
//     }
//     {
//         std::vector<int> iter = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
//         11,
//                                  12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
//                                  23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
//                                  34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
//                                  45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
//                                  56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
//                                  67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
//                                  78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88,
//                                  89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99};

//         int sm = itertools::sum<int>(iter);

//         assert(sm == 4950);
//     }
// }

// void
// time_multiple_tests()
// {
//     size_t N = 1'000;
//     {
//         size_t M = 10'000;

//         auto func1 = [&]() {
//             size_t t = 0;
//             std::string h = "";
//             for (auto i : itertools::range(M)) {
//                 h = std::to_string(i);
//             };
//             return t;
//         };

//         auto func2 = [&]() {
//             size_t t = 0;
//             std::string h = "";
//             for (size_t i = 0; i < M; i++) {
//                 h = std::to_string(i);
//             };
//             return t;
//         };

//         auto [times, extremal_times] =
//           itertools::time_multiple(N, func1, func2);
//         for (auto [key, value] : times) {
//             fmt::print("function {}:\n", key);
//             for (auto i : extremal_times[key]) {
//                 fmt::print("\t{}\n", i);
//             }
//         }
//     }
//     {
//         size_t M = 10'000;

//         auto func1 = [&]() {
//             std::vector<int> iv1(M, 1);
//             std::string h = "";
//             for (auto [n, i] : itertools::enumerate(iv1)) {
//                 h = std::to_string(n);
//                 h = std::to_string(i);
//             };
//         };

//         auto func2 = [&]() {
//             std::vector<int> iv1(M, 1);
//             std::string h = "";
//             for (auto [n, i] :
//                  itertools::zip(itertools::range(iv1.size()), iv1)) {
//                 h = std::to_string(n);
//                 h = std::to_string(i);
//             };
//         };

//         auto func3 = [&]() {
//             std::vector<int> iv1(M, 1);
//             std::string h = "";
//             size_t n = 0;
//             for (auto i : iv1) {
//                 h = std::to_string(n);
//                 h = std::to_string(i);
//                 n++;
//             };
//         };

//         auto [times, extremal_times] =
//           itertools::time_multiple(N, func1, func2, func3);
//         for (auto [key, value] : times) {
//             fmt::print("function {}:\n", key);
//             for (auto i : extremal_times[key]) {
//                 fmt::print("\t{}\n", i);
//             }
//         }
//     }
// }

// void
// to_string_tests(bool print = false)
// {
//     {
//         auto iter = std::make_tuple(1, 2, 3, 4, 5, 6);
//         auto ndim = itertools::get_ndim(iter);
//         std::string s = itertools::to_string(iter, [](auto&& v) ->
//         std::string {
//             return std::to_string(v);
//         });
//         if (print) {
//             std::cout << s << std::endl;
//         }
//     }
//     {
//         auto iter = std::make_tuple(std::make_tuple(1, 2, 3, 4, 5, 6),
//                                     std::make_tuple(1, 2, 3, 4, 5, 6),
//                                     std::make_tuple(1, 2, 3, 4, 5, 6));
//         auto ndim = itertools::get_ndim(iter);
//         std::string s = itertools::to_string(iter, [](auto&& v) ->
//         std::string {
//             return " " + std::to_string(v) + " ";
//         });
//         if (print) {
//             std::cout << s << std::endl;
//         }
//     }
//     {
//         std::vector<std::map<int, int>> iter = {{{1, 2}, {3, 4}},
//                                                 {{5, 6}, {7, 8}}};
//         auto ndim = itertools::get_ndim(iter);
//         std::string s = itertools::to_string(iter, [](auto&& v) ->
//         std::string {
//             return std::to_string(v);
//         });
//         if (print) {
//             std::cout << s << std::endl;
//         }
//     }
//     {
//         std::vector<std::tuple<std::vector<std::vector<std::vector<int>>>,
//         int>>
//           iter = {{{{{1, 2}}, {{3, 4}}}, 1}, {{{{5, 6}}, {{7, 8}}}, 4}};
//         auto ndim = itertools::get_ndim(iter);
//         std::string s = itertools::to_string(iter, [](auto&& v) ->
//         std::string {
//             return std::to_string(v);
//         });
//         if (print) {
//             std::cout << s << std::endl;
//         }
//     }
//     {
//         std::vector<
//           std::list<std::vector<std::list<std::vector<std::deque<int>>>>>>
//           iter = {{{{{{0, 1}, {2, 3}},

//                      {{4, 5}, {6, 7}}},

//                     {{{8, 9}, {10, 11}},

//                      {{12, 13}, {14, 15}}}},

//                    {{{{16, 17}, {18, 19}},

//                      {{20, 21}, {22, 23}}},

//                     {{{24, 25}, {26, 27}},

//                      {{28, 29}, {30, 31}}}}},

//                   {{{{{32, 33}, {34, 35}},

//                      {{36, 37}, {38, 39}}},

//                     {{{40, 41}, {42, 43}},

//                      {{44, 45}, {46, 47}}}},

//                    {{{{48, 49}, {50, 51}},

//                      {{52, 53}, {54, 55}}},

//                     {{{56, 57}, {58, 59}},

//                      {{60, 61}, {62, 63}}}}}};
//         auto ndim = itertools::get_ndim(iter);
//         std::string s = itertools::to_string(iter, [](auto&& v) ->
//         std::string {
//             return std::to_string(v);
//         });
//         if (print) {
//             std::cout << s << std::endl;
//         }
//     }
//     {
//         std::vector<std::tuple<std::list<std::vector<std::vector<int>>>,
//                                int,
//                                std::map<int, std::tuple<int, int, int>>>>
//           iter =
//             {{{{{1, 2}}, {{3, 4}}},
//               1,
//               {{1, {0, 1, 2}}, {2, {1, 2, 3}}, {3, {2, 3, 4}}, {4, {3, 4,
//               5}}}},
//              {{{{5, 6}}, {{7, 8}}},
//               4,
//               {{1, {0, 1, 2}},
//                {2, {1, 2, 3}},
//                {3, {2, 3, 4}},
//                {4, {3, 4, 5}}}}};
//         auto ndim = itertools::get_ndim(iter);
//         std::string s = itertools::to_string(iter, [](auto&& v) ->
//         std::string {
//             return std::to_string(v);
//         });
//         if (print) {
//             std::cout << s << std::endl;
//         }
//     }
//     {
//         std::vector<std::vector<int>> iter =
//           {{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
//           16,
//             17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
//             33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
//             49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
//             65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
//             81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96,
//             97, 98, 99},
//            {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
//            16,
//             17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
//             33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
//             49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
//             65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
//             81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96,
//             97, 98, 99}};
//         auto ndim = itertools::get_ndim(iter);
//         std::string s = itertools::to_string(iter, [](auto&& v) ->
//         std::string {
//             return std::to_string(v);
//         });
//         if (print) {
//             std::cout << s << std::endl;
//         }
//     }
// }

void
print_pow_2_tables(int n)
{
    std::cout << "[";
    for (auto i : itertools::range(n)) {
        uint64_t nn = pow(2, 64 - (i + 1));
        fmt::print("{:d},", nn);
    }
    std::cout << "]\n";
    std::cout << "[";
    for (auto i : itertools::range(n)) {
        double nn = pow(2, pow(2, -i));
        fmt::print("{:.52f},", nn);
    }
    std::cout << "]\n";
}

constexpr uint64_t two53 = 9007199254740992 * 2;

constexpr double log2_e = 1.442695040888963407359924681001892137426645954152;

union double_bits
{
    double dvalue;
    uint64_t ivalue;

    constexpr double_bits(double d)
      : dvalue(d){};
};

union doublebits
{
    double d;
    constexpr doublebits(double dd)
      : d(dd){};
    struct
    {

        uint64_t mant : 52;
        uint expo : 11;
        uint sign : 1;

    } parts;
};

constexpr std::tuple<uint, int, uint64_t>
myfrexp(double x)
{
    uint64_t ivalue = 0;

    double_bits db(x);

    uint sign = db.ivalue >> 63;
    int exponent = (db.ivalue << 1) >> 53;
    uint64_t mantissa = db.ivalue & 0xfffffffffffffL;

    if (exponent == 0) {
        exponent++;
    } else {
        mantissa |= (1L << 52);
    }
    exponent -= 1022;

    return {sign, exponent, mantissa};
}

double constexpr myexp(double x)
{
    bool inverse = false;
    if (x < 0) {
        inverse = true;
        x = -1 * x;
    }
    double z = x * log2_e;

    uint64_t n = z;

    double m = z - n;

    auto [sign, exponent, mantissa] = myfrexp(m);

    // int exponent;
    // double mm = frexp(m, &exponent);
    // uint64_t mantissa = static_cast<uint64_t>(mm * two53);

    double q = 1.0;

    if (n < 64) {
        q *= POW_2[64 - (n + 1)];
    } else {
        return -1;
    }

    size_t i = 0;
    size_t high_bit = 0;

    while (i < 64) {
        if (mantissa & POW_2[i]) {
            if (!high_bit) {
                high_bit = i - 1 + exponent;
            }
            q *= POW_2_M2[i - high_bit];
        }
        i++;
    }

    return inverse ? 1 / q : q;
}

void
frexp_tests(double d)
{
    {
        auto [s, e, m] = myfrexp(d);
        fmt::print("{}, {}\n", e, static_cast<double>(m) / two53);
    }

    {
        doublebits db(d);
        fmt::print("{}, {}\n",
                   ((int) db.parts.expo) - 1022,
                   static_cast<double>(db.parts.mant) / two53 + 0.5);
    }
    {
        uint64_t ivalue = *((uint64_t*) (&d));

        double realMant = 1.0;
        int exponent = (int) ((ivalue >> 52) & 0x7ffL);
        uint64_t mantissa = ivalue & 0xfffffffffffffL;

        uint64_t m = mantissa;

        if (exponent == 0) {
            exponent++;
        } else {
            mantissa = mantissa | (1L << 52);
        }

        // bias the exponent - actually biased by 1023.
        // we are treating the mantissa as m.0 instead of 0.m
        //  so subtract another 52.
        exponent -= 1075;
        realMant = mantissa;

        // normalize
        while (realMant >= 1.0) {
            mantissa >>= 1;
            realMant /= 2.;
            exponent++;
        }

        // if (neg) {
        //     realMant = realMant * -1;
        // }

        fmt::print("{}, {}\n", exponent, realMant);
    }
    {
        int e;
        double t_m = frexp(d, &e);
        uint64_t m = static_cast<uint64_t>(t_m * two53);

        fmt::print("{}, {}\n", e, t_m);
    }
}

int
main()
{

    for (int i : itertools::range(45)) {
        // double d = 26 * i + i / 10.0f + i / 7.f + 999.0f / i;
        // frexp_tests(d);
        // std::cout << std::endl;
        fmt::print("{}\n", i);
        fmt::print("{:.52f}\n", myexp(-i));
        fmt::print("{:.52f}\n", exp(-i));
    }

    // print_pow_2_tables(64);
    // zip_tests();
    // any_tests();
    // enumerate_tests();
    // range_tests();
    // rvalue_zip_tests();
    // itertools_tests();
    // tupletools_tests();
    // reduction_tests();
    // generator_tests();
    // time_multiple_tests();
    // to_string_tests();

    fmt::print("tests complete\n");
    return 0;
}