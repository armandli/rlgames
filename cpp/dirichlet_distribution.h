#ifndef RLGAMES_DIRICHLET_DISTRIBUTION
#define RLGAMES_DIRICHLET_DISTRIBUTION

#include <random>
#include <array>
#include <initializer_list>
#include <algorithm>

#include <type_alias.h>

namespace rlgames {

namespace s = std;

template <uint Size, typename RealType = float>
struct dirichlet_distribution {
  using ty = RealType;

  class param_type {
    s::array<ty, Size> alphas;
  public:
    using distribution_type = dirichlet_distribution;
    explicit param_type(s::initializer_list<ty> as){
      s::copy(s::begin(as), s::end(as), s::begin(alphas));
    }

    ty param(uint idx) const {
      return alphas[idx];
    }
    ty operator[](uint idx) const {
      return alphas[idx];
    }

    bool operator==(const param_type& o) const {
      for (unsigned i = 0; i < Size; ++i)
        if (alphas[i] != o.alphas[i])
          return false;
      return true;
    }
    bool operator!=(const param_type& o) const {
      return not (*this == o);
    }
  };

  dirichlet_distribution(){
    for (uint i = 0; i < Size; ++i)
      mGammas[i] = s::gamma_distribution<ty>(DFV, DFV);
  }
  explicit dirichlet_distribution(ty alpha){
    for (uint i = 0; i < Size; ++i)
      mGammas[i] = s::gamma_distribution<ty>(alpha, DFV);
  }
  explicit dirichlet_distribution(s::initializer_list<ty> params){
    param_type ps(params);
    for (uint i = 0; i < Size; ++i)
      mGammas[i] = s::gamma_distribution<ty>(ps[i], DFV);
  }

  void param(const param_type& p){
    for (uint i = 0; i < Size; ++i)
      mGammas[i] = s::gamma_distribution<ty>(p[i], DFV);
  }

  template <class Generator>
  s::array<ty, Size> operator()(Generator& g){
    return generate(g);
  }
  template <class Generator>
  s::array<ty, Size> operator()(Generator& g, const param_type& params){
    param(params);
    return generate(g);
  }

  ty min() const { return 0.; }
  ty max() const { return 1.; }

  bool operator==(const dirichlet_distribution<Size, ty>& o) const {
    for (uint i = 0; i < Size; ++i)
      if (mGammas[i] != o.mGammas[i])
        return false;
    return true;
  }
  bool operator!=(const dirichlet_distribution<Size, ty>& o) const {
    return not (*this == o);
  }

private:
  s::array<s::gamma_distribution<ty>, Size> mGammas;
  static constexpr RealType DFV = 1.;

  template <typename G>
  s::array<ty, Size> generate(G& g){
    s::array<ty, Size> ret;
    for (uint i = 0; i < Size; ++i)
      ret[i] = mGammas[i](g);
    ty sum = s::accumulate(s::begin(ret), s::end(ret), (ty)0.);
    s::for_each(s::begin(ret), s::end(ret), [&sum](ty& v){ v /= sum; });

    return ret;
  }
};

} // rlgames

#endif//RLGAMES_DIRICHLET_DISTRIBUTION
