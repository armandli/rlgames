#ifndef GRIDWORLD_LEARNING_DEBUG
#define GRIDWORLD_LEARNING_DEBUG

#include <torch/torch.h>

#include <cassert>
#include <type_alias.h>
#include <iostream>

namespace gridworld_pt {

namespace t = torch;
namespace s = std;

void check_dim(t::Tensor t, init_list<sint64> expected_dims){
  assert(t.dim() == (sint64)expected_dims.size());
  uint i = 0;
  for (int expected : expected_dims){
    assert(t.size(i) == expected);
    i++;
  }
}

void print_dim(t::Tensor t){
  s::cout << "[";
  for (uint i = 0; i < t.dim(); ++i){
    s::cout << t.size(i) << ", ";
  }
  s::cout << "]" << s::endl;
}

} //gridworld_pt

#endif//GRIDWORLD_LEARNING_DEBUG
