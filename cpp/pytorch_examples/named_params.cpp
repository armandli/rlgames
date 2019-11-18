#include <torch/torch.h>

#include <iostream>

namespace t = torch;
namespace s = std;

class TestModelImpl : public t::nn::Module {
  t::nn::Linear l1, l2;
  t::Tensor     b1, b2;
public:
  TestModelImpl(unsigned input, unsigned hidden, unsigned output):
    l1(register_module("l1", t::nn::Linear(input, hidden))),
    l2(register_module("l2", t::nn::Linear(hidden, output))),
    b1(register_parameter("b1", t::randn(hidden))),
    b2(register_parameter("b2", t::randn(output)))
  {}

  t::Tensor forward(t::Tensor x){
    x = t::relu(l1(x) + b1);
    x = t::relu(l2(x) + b2);
    return x;
  }
};
TORCH_MODULE(TestModel);

int main(){
  TestModel model(16, 32, 4);
  auto params = model->named_parameters(true);
  auto buffers = model->named_buffers(true);

  s::cout << "Params:" << s::endl;
  for (auto& item : params){
    s::cout << item.key() << s::endl;
  }
  s::cout << "Buffers:" << s::endl;
  for (auto& item : buffers){
    s::cout << item.key() << s::endl;
  }

  auto param_vec = model->parameters(true);
  s::cout << "Param Size: " << param_vec.size() << s::endl;
}
