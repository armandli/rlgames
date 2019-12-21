#include <learning_util.h>

#include <torch/torch.h>

#include <iostream>
#include <vector>

namespace t = torch;
namespace s = std;
namespace m = gridworld_pt;

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
  TestModel model1(4, 8, 4);
  TestModel model2(4, 8, 4);

  m::copy_state(model2, model1);

  s::vector<t::Tensor> params1 = model1->parameters(true);
  s::vector<t::Tensor> params2 = model2->parameters(true);

  s::cout << "params1:" << s::endl;
  for (t::Tensor& ts : params1)
    s::cout << ts << s::endl;

  s::cout << "params2:" << s::endl;
  for (t::Tensor& ts : params2)
    s::cout << ts << s::endl;

  t::optim::Adam optimizer(model1->parameters(), t::optim::AdamOptions(1e-1F));

  float ax[] = {0.5, 0.4, 0.9, 0.1};
  float ay[] = {1.,  0.,  0.,  0.};

  t::Tensor input = t::from_blob(ax, {4});
  t::Tensor target = t::from_blob(ay, {4});

  for (int i = 0; i < 1000; ++i){
    optimizer.zero_grad();
    t::Tensor output = model1->forward(input);
    t::Tensor loss = t::mse_loss(output, target.detach());

    s::cout << "loss " << loss.item().to<float>() << s::endl;

    loss.backward();
    optimizer.step();
  }

  s::cout << "After: " << s::endl;
  s::cout << "params1:" << s::endl;
  for (t::Tensor& ts : params1)
    s::cout << ts << s::endl;

  s::cout << "params2:" << s::endl;
  for (t::Tensor& ts : params2)
    s::cout << ts << s::endl;

}
