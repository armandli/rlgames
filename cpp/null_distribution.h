#ifndef RLGAMES_NULL_DISTRIBUTION
#define RLGAMES_NULL_DISTRIBUTION

#include <vector>
#include <algorithm>

#include <type_alias.h>

namespace rlgames {

namespace s = std;

//placeholder distribution for generating a vector
//of random values, null distribution generates only 0

template <uint Size, typename RealType = float>
struct null_distribution {
  using ty = RealType;

  class param_type {
  public:
    using distribution_type = null_distribution;
    explicit param_type(s::initializer_list<ty>){}

    ty param(uint) const {
      return 0.F;
    }
    ty operator[](uint) const {
      return 0.F;
    }

    bool operator==(const param_type&) const {
      return true;
    }
    bool operator!=(const param_type&) const {
      return false;
    }
  };

  null_distribution() = default;
  explicit null_distribution(ty){}
  explicit null_distribution(s::initializer_list<ty>){}

  template <class Generator>
  s::array<ty, Size> operator()(Generator&){
    return generate();
  }
  template <class Generator>
  s::array<ty, Size> operator()(Generator&, const param_type&){
    return generate();
  }

  ty min() const { return 0.; }
  ty max() const { return 0.; }

  bool operator==(const null_distribution<Size, ty>&) const {
    return true;
  }
  bool operator!=(const null_distribution<Size, ty>&) const {
    return false;
  }
private:
  s::array<ty, Size> generate(){
    s::array<ty, Size> ret;
    s::fill(s::begin(ret), s::end(ret), 0.F);
    return ret;
  }
};

} // rlgames

#endif//RLGAMES_NULL_DISTRIBUTION
