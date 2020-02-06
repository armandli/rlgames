#include <gtest/gtest.h>

#include <torch/torch.h>

#include <vector>

#include <gridworld.h>
#include <gridworld_models.h>

namespace s = std;
namespace g = gridworld;
namespace t = torch;
namespace m = gridworld_pt;

TEST(TestDistributionalQModel, TestDistributionalQModel1){
  t::Device device(t::kCPU);
  uint size = 4;
  m::GridEnv env(size, m::GridEnvMode::StaticSimple);
  m::GridStateEncoder encoder(env);
  m::GridActionEncoder action_encoder(env);
  m::SimpleDistQModel model(encoder.state_size().flatten_size(), 164, 150, action_encoder.action_size(), 51);

  g::GridWorld ins = env.create();
  t::Tensor t = encoder.encode_state(ins.get_state(), device);
  t::Tensor adists = model->forward(t);
}
