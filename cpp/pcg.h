#ifndef RLGAMES_PCG
#define RLGAMES_PCG

#include <cstdint>
#include <random>

#include <type_alias.h>

namespace rlgames {

class Pcg {
public:
  using result_type = uint;
  static constexpr result_type (min)() { return 0; }
  static constexpr result_type (max)() { return UINT32_MAX; }


  Pcg(): mState(0x853C49E6748FEA9BULL), mInc(0xDA3E39CB94B95BDBULL){}
  explicit Pcg(std::random_device &rd){
    seed(rd);
  }
  template <typename T>
  explicit Pcg(const T& s): mState(0x853C49E6748FEA9BULL), mInc(0xDA3E39CB94B95BDBULL) {
    discard(s);
  }

  void seed(std::random_device &rd){
    uint64_t s0 = uint64_t(rd()) << 31 | uint64_t(rd());
    uint64_t s1 = uint64_t(rd()) << 31 | uint64_t(rd());

    mState = 0;
    mInc = (s1 << 1) | 1;
    operator()();
    mState += s0;
    operator()();
  }

  result_type operator()(){
    uint64_t oldstate = mState;
    mState = oldstate * 6364136223846793005ULL + mInc;
    uint32_t xorshifted = uint32_t(((oldstate >> 18u) ^ oldstate) >> 27u);
    int rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
  }

  void discard(unsigned long long n){
    for (unsigned long long i = 0; i < n; ++i)
      operator()();
  }

  bool operator==(const Pcg& o) const {
    return mState == o.mState && mInc == o.mInc;
  }
  bool operator!=(const Pcg& o) const {
    return not this->operator==(o);
  }
private:
  uint64_t mState;
  uint64_t mInc;
};

} // rlgames


#endif//RLGAMES_PCG
