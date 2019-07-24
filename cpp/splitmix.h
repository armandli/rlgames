#ifndef RLGAMES_SPLITMIX
#define RLGAMES_SPLITMIX

#include <random>

#include <type_alias.h>

namespace rlgames {

class Splitmix {
public:
  using result_type = uint;
  static constexpr result_type (min)() { return 0; }
  static constexpr result_type (max)() { return UINT32_MAX; }

  Splitmix() : mSeed(1) {}
  explicit Splitmix(std::random_device &rd){
    seed(rd);
  }
  template <typename T>
  explicit Splitmix(const T& s): mSeed(s) {}

  void seed(std::random_device &rd){
    mSeed = uint64_t(rd()) << 31 | uint64_t(rd());
  }

  result_type operator()(){
    uint64_t z = (mSeed += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return result_type((z ^ (z >> 31)) >> 31);
  }

  void discard(unsigned long long n){
    for (unsigned long long i = 0; i < n; ++i)
      operator()();
  }

  bool operator==(const Splitmix& o) const {
    return mSeed == o.mSeed;
  }
  bool operator!=(const Splitmix& o) const {
    return not this->operator==(o);
  }
private:
  uint64_t mSeed;
};

} // rlgames

#endif//RLGAMES_SPLITMIX
