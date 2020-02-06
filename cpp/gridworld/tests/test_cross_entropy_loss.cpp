#include <iostream>

#include <torch/torch.h>

//test if custom implementation of cross entropy loss is correct

namespace s = std;
namespace t = torch;

void test_cross_entropy1(){
  float target_vec[] = {
    0.05, 0.05, 0.12, 0.09, 0.01, 0.17, 0.03, 0.08, 0.2, 0.2,
  };
  float input_vec[] = {
    0.12, 0.09, 0.15, 0.25, 0.01, 0.17, 0.05, 0.06, 0.05, 0.05,
  };
  t::Tensor target = t::from_blob(target_vec, {1, 10});
  t::Tensor input  = t::from_blob(input_vec, {1, 10});

  //does not work, target tensor needs to be the label in one dimension, not a distribution
  //t::nn::CrossEntropyLoss loss;

  //t::Tensor expected = loss->forward(input, target);
  t::Tensor calculated = t::mean(t::sum(-1.F * target * t::log(input), -1));

  //s::cout << "Expected:" << s::endl;
  //s::cout << expected << s::endl;
  s::cout << "Calculated:" << s::endl;
  s::cout << calculated << s::endl;
}

void test_cross_entropy2(){
  float target_vec[] = {
    0.05, 0.05, 0.12, 0.09, 0.01, 0.17, 0.03, 0.08, 0.2, 0.2,
  };
  float input_vec[] = {
    0.05, 0.05, 0.12, 0.09, 0.01, 0.17, 0.03, 0.08, 0.2, 0.2,
  };
  t::Tensor target = t::from_blob(target_vec, {1, 10});
  t::Tensor input  = t::from_blob(input_vec, {1, 10});

  //does not work, target tensor needs to be the label in one dimension, not a distribution
  //t::nn::CrossEntropyLoss loss;

  //t::Tensor expected = loss->forward(input, target);
  t::Tensor calculated = t::mean(t::sum(-1.F * target * t::log(input), -1));

  //s::cout << "Expected:" << s::endl;
  //s::cout << expected << s::endl;
  s::cout << "Calculated:" << s::endl;
  s::cout << calculated << s::endl;
}


int main(){
  test_cross_entropy1();
  test_cross_entropy2();
}
