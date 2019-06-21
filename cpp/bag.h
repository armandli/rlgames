#ifndef RLGAMES_BAG
#define RLGAMES_BAG

#include <cassert>
#include <vector>

namespace s = std;

namespace rlgames {

template <typename E>
class bag {
  s::vector<E> mVec;

  using ref = E&;
  using cref = const E&;
  using rref = E&&;
  using iter = typename s::vector<E>::iterator;
  using citer = typename s::vector<E>::const_iterator;

public:
  bag() = default;
  bag(const bag& o) = default;
  bag& operator=(const bag& o){
    //TODO: proper implementation
    mVec = o.mVec;
    return *this;
  }

  //capacity
  size_t size() const { return mVec.size(); }
  void reserve(size_t sz){ mVec.reserve(sz); }
  bool empty() const { return mVec.empty(); }
  
  //element access
  ref operator[](size_t i){
    assert(i < mVec.size());
    return mVec[i];
  }
  cref operator[](size_t i) const {
    assert(i < mVec.size());
    return mVec[i];
  }
  ref at(size_t i){ return mVec.at(i); }
  cref at(size_t i) const { return mVec.at(i); }
  ref back(){ return mVec.back(); }
  cref back() const { return mVec.back(); }
  ref front(){ return mVec.front(); }
  cref front() const { return mVec.front(); }

  //iterator
  iter begin(){ return mVec.begin(); }
  iter end(){ return mVec.end(); }
  citer cbegin() const { return mVec.cbegin(); }
  citer cend() const { return mVec.cend(); }

  //modifiers
  void push_back(const E& v){ mVec.push_back(v); }
  void pop_back(){ mVec.pop_back(); }
  void clear() noexcept { mVec.clear(); }

  ref push(size_t idx, cref value){
    assert(idx <= mVec.size());

    if (idx != mVec.size()){
      mVec.push_back(mVec[idx]);
      mVec[idx] = value;
      return mVec[idx];
    } else {
      mVec.push_back(value);
      return mVec.back();
    }
  }
  //TODO: don't return anything
  E pop(size_t idx){
    assert(idx < mVec.size());

    E copy = mVec[idx];
    if (mVec.size() > 1){
      mVec[idx] = mVec.back();
      mVec.pop_back();
    } else {
      mVec.pop_back();
    }
    return copy;
  }

  template <class... Args>
  void emplace_back(Args&&... args){
    mVec.emplace_back(args...);
  }
};

template <typename E>
typename s::vector<E>::iterator begin(bag<E>& b){
  return b.begin();
}

template <typename E>
typename s::vector<E>::iterator end(bag<E>& b){
  return b.end();
}

} // rlgames

#endif//RLGAMES_BAG
