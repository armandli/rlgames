#include <cassert>
#include <ctime>
#include <random>
#include <iostream>

namespace s = std;

template <typename ENG>
unsigned sample_discrete_distribution(float* probs, size_t sz, ENG& eng){
  s::discrete_distribution<uint> dist(probs, probs + sz);
  return dist(eng);
}

int main(){
  srand(time(NULL));

  s::default_random_engine eng(time(NULL));

  constexpr size_t size = 4;
  float p[size] = {0.1, 0.1, 0.7, 0.1};

  unsigned values[size] = {0};
  for (int i = 0; i < 1000; ++i){
    unsigned idx = sample_discrete_distribution(p, size, eng);
    assert(idx < 4);
    values[idx]++;
  }

  for (unsigned i = 0; i < size; ++i){
    s::cout << i << ": ";
    for (unsigned j = 0; j < values[i] / 10; ++j)
      s::cout << "*";
    s::cout << "\n";
  }
}
