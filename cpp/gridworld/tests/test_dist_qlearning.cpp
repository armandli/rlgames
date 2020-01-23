#include <iostream>

#include <torch/torch.h>

#include <type_alias.h>

namespace s = std;
namespace t = torch;

class TestModelImpl : public t::nn::Module {
  t::nn::Linear l1, l2, l3;
  sint64 action_sz;
public:
  TestModelImpl(sint64 isz, sint64 l1sz, sint64 l2sz, sint64 action_sz, sint64 distout_sz):
    l1(register_module("l1", t::nn::Linear(isz, l1sz))),
    l2(register_module("l2", t::nn::Linear(l1sz, l2sz))),
    l3(register_module("l3", t::nn::Linear(l2sz, action_sz * distout_sz))),
    action_sz(action_sz)
  {}
  t::Tensor forward(t::Tensor x){
    x = t::relu(l1(x));
    x = t::relu(l2(x));
    x = l3(x);
    if (x.dim() == 1)
      x = x.reshape({action_sz, -1});
    else
      x = x.reshape({x.size(0), action_sz, -1});
    x = t::softmax(x, -1);
    return x;
  }
};
TORCH_MODULE(TestModel);

void test_model_terminal1(){
  float learning_rate = 1e-2F;
  TestModel model(16, 12, 8, 4, 9);
  t::optim::Adam optimizer(model->parameters(), t::optim::AdamOptions(learning_rate));
  float input_vec[] = {
    0.6, 0.3, 0.2, 1.,
    0.1, 0.2, 0.3, 0.4,
    0.9, 0.8, 0.7, 0.6,
    0.5, 0.5, 0.5, 0.5,
  };
  t::Tensor input = t::from_blob(input_vec, {16});

  float target_dev[] = {
    0.F, 0.F, 0.F, 0.F, 1.F, 0.F, 0.F, 0.F, 0.F,
  };
  t::Tensor target = t::from_blob(target_dev, {1, 9});

  long select_vec[] = {1};
  t::Tensor select = t::from_blob(select_vec, {1}, t::kLong);
  select = select.repeat_interleave(9).reshape({1, 1, 9});

  s::cout << select << s::endl;

  for (uint i = 0; i < 100; ++i){
    model->zero_grad();

    t::Tensor out = model->forward(input);
    out = out.reshape({1, 4, 9});

    out = t::squeeze(t::gather(out, 1, select));

    t::Tensor loss_dev = t::mean(t::sum(-1.F * target.detach() * t::log(out), -1));
    loss_dev.backward();
    optimizer.step();
  }

  t::Tensor gout = model->forward(input);
  gout = gout.reshape({1, 4, 9});
  gout = t::squeeze(t::gather(gout, 1, select));
  s::cout << "Test Terminal 1" << s::endl;
  s::cout << gout << s::endl;
}

void test_partial(){

}

int main(){
  test_model_terminal1();
}
