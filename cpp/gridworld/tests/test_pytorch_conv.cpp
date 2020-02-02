#include <torch/torch.h>

#include <iostream>
#include <vector>

#include <type_alias.h>

namespace t = torch;
namespace s = std;

class ModelImpl : public t::nn::Module {
  t::nn::Conv2d cl1;
  t::nn::Linear l2;
public:
  ModelImpl(unsigned ifsz, unsigned l1fsz, unsigned l1ksz, unsigned osz):
    cl1(register_module("cl1", t::nn::Conv2d(t::nn::Conv2dOptions(ifsz, l1fsz, l1ksz)))),
    l2(register_module("l2", t::nn::Linear((6 - l1ksz + 1) * (6 - l1ksz + 1) * 8, osz)))
  {}
  t::Tensor forward(t::Tensor x){
    x = t::relu(cl1(x));
    x = t::softmax(l2(x.flatten()), -1);
    return x;
  }
};
TORCH_MODULE(Model);

int main(){
  Model m1(2, 8, 3, 4);

  float input_vec[] = {
    1,2,3,4,5,6,
    1,2,3,4,5,6,
    1,2,3,4,5,6,
    1,2,3,4,5,6,
    1,2,3,4,5,6,
    1,2,3,4,5,6,

    0.1,0.2,0.3,0.4,0.5,0.6,
    0.1,0.2,0.3,0.4,0.5,0.6,
    0.1,0.2,0.3,0.4,0.5,0.6,
    0.1,0.2,0.3,0.4,0.5,0.6,
    0.1,0.2,0.3,0.4,0.5,0.6,
    0.1,0.2,0.3,0.4,0.5,0.6,
  };
  t::Tensor input = t::from_blob(input_vec, {1, 2, 6, 6});
  t::Tensor out = m1->forward(input);

  s::cout << out << s::endl;
}
