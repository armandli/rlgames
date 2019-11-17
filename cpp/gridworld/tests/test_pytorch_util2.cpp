#include <pytorch_util.h>

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

  t::Device cpu_device(t::kCPU);
  t::Device device(t::kCPU);
  if (t::cuda::is_available()){
    s::cout << "Using CUDA" << s::endl;
    device = t::Device(t::kCUDA);
  }

  model1->to(device);
  model2->to(device);

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
  t::Tensor input_dev = input.to(device);
  t::Tensor target = t::from_blob(ay, {4});
  t::Tensor target_dev = target.to(device);

  for (int i = 0; i < 1000; ++i){
    model1->zero_grad();
    t::Tensor output_dev = model1->forward(input_dev);
    t::Tensor loss_dev = t::mse_loss(output_dev, target_dev.detach());
    t::Tensor loss = loss_dev.to(cpu_device);

    //s::cout << "loss " << loss.item().to<float>() << s::endl;

    loss_dev.backward();
    optimizer.step();
  }

  s::cout << "After: " << s::endl;
  s::cout << "params1:" << s::endl;
  for (t::Tensor& ts : params1)
    s::cout << ts << s::endl;

  s::cout << "params2:" << s::endl;
  for (t::Tensor& ts : params2)
    s::cout << ts << s::endl;

  m::copy_state(model2, model1);
  s::cout << "After Copy: " << s::endl;
  s::cout << "params1:" << s::endl;
  for (t::Tensor& ts : params1)
    s::cout << ts << s::endl;

  s::cout << "params2:" << s::endl;
  for (t::Tensor& ts : params2)
    s::cout << ts << s::endl;
}
