#include <gtest/gtest.h>

#include <torch/torch.h>

#include <gridworld.h>
#include <gridworld_models.h>

namespace s = std;
namespace g = gridworld;
namespace t = torch;
namespace m = gridworld_pt;

TEST(TestGridStateEncoder, TestEncode1){
  t::Device device(t::kCPU);
  uint size = 4;
  m::GridEnv env(size, m::GridEnvMode::StaticSimple);
  m::GridStateEncoder encoder(env);

  g::GridWorld ins = env.create();
  t::Tensor t = encoder.encode_state(ins.get_state(), device);
  EXPECT_EQ(1, t.dim());
  EXPECT_EQ(64, t.sizes()[0]);

  for (uint i = 0; i < size; ++i)
    for (uint j = 0; j < size; ++j){
      if (ins.get_state().get(g::Pt(i, j)) == g::Obj::Wall){
        EXPECT_EQ(1.f, t[size * size + i * size + j].item().to<float>());
      } else {
        EXPECT_EQ(0.f, t[size * size + i * size + j].item().to<float>());
      }
      if (ins.get_state().get(g::Pt(i, j)) == g::Obj::Player){
        EXPECT_EQ(1.f, t[i * size + j].item().to<float>());
      } else {
        EXPECT_EQ(0.f, t[i * size + j].item().to<float>());
      }
      if (ins.get_state().get(g::Pt(i, j)) == g::Obj::Sink){
        EXPECT_EQ(1.f, t[size * size * 2 + i * size + j].item().to<float>());
      } else {
        EXPECT_EQ(0.f, t[size * size * 2 + i * size + j].item().to<float>());
      }
      if (ins.get_state().get(g::Pt(i, j)) == g::Obj::Goal){
        EXPECT_EQ(1.f, t[size * size * 3 + i * size + j].item().to<float>());
      } else {
        EXPECT_EQ(0.f, t[size * size * 3 + i * size + j].item().to<float>());
      }
    }
}

TEST(TestGridStateEncoder, TestEncode2){
  t::Device device(t::kCPU);
  uint size = 8;
  m::GridEnv env(size, m::GridEnvMode::StaticSimple);
  m::GridStateEncoder encoder(env);

  g::GridWorld ins = env.create();
  t::Tensor t = encoder.encode_state(ins.get_state(), device);
  EXPECT_EQ(1, t.dim());
  EXPECT_EQ(256, t.sizes()[0]);

  for (uint i = 0; i < size; ++i)
    for (uint j = 0; j < size; ++j){
      if (ins.get_state().get(g::Pt(i, j)) == g::Obj::Wall){
        EXPECT_EQ(1.f, t[size * size + i * size + j].item().to<float>());
      } else {
        EXPECT_EQ(0.f, t[size * size + i * size + j].item().to<float>());
      }
      if (ins.get_state().get(g::Pt(i, j)) == g::Obj::Player){
        EXPECT_EQ(1.f, t[i * size + j].item().to<float>());
      } else {
        EXPECT_EQ(0.f, t[i * size + j].item().to<float>());
      }
      if (ins.get_state().get(g::Pt(i, j)) == g::Obj::Sink){
        EXPECT_EQ(1.f, t[size * size * 2 + i * size + j].item().to<float>());
      } else {
        EXPECT_EQ(0.f, t[size * size * 2 + i * size + j].item().to<float>());
      }
      if (ins.get_state().get(g::Pt(i, j)) == g::Obj::Goal){
        EXPECT_EQ(1.f, t[size * size * 3 + i * size + j].item().to<float>());
      } else {
        EXPECT_EQ(0.f, t[size * size * 3 + i * size + j].item().to<float>());
      }
    }
}

TEST(TestGridStateEncoder, TestDecode1){
  t::Device device(t::kCPU);
  uint size = 8;
  m::GridEnv env(size, m::GridEnvMode::StaticSimple);
  m::GridStateEncoder encoder(env);

  g::GridWorld ins = env.create();
  t::Tensor t = encoder.encode_state(ins.get_state(), device);
  g::GridState st = encoder.decode_state(t);

  for (uint i = 0; i < size * size; ++i)
    EXPECT_EQ(ins.get_state().get(i), st.get(i));
}

TEST(TestGridActionEncoder, TestEncode1){
  t::Device device(t::kCPU);
  m::GridEnv env(4, m::GridEnvMode::StaticSimple);
  m::GridActionEncoder encoder(env);

  t::Tensor t1 = encoder.encode_action(g::Action::UP, device);
  EXPECT_EQ(1, t1.dim());
  EXPECT_EQ(4, t1.sizes()[0]);
  EXPECT_EQ(1.f, t1[0].item().to<float>());
  EXPECT_EQ(0.f, t1[1].item().to<float>());

  t::Tensor t2 = encoder.encode_action(g::Action::DN, device);
  EXPECT_EQ(1, t2.dim());
  EXPECT_EQ(4, t2.sizes()[0]);
  EXPECT_EQ(1.f, t2[1].item().to<float>());
  EXPECT_EQ(0.f, t2[0].item().to<float>());

  t::Tensor t3 = encoder.encode_action(g::Action::LF, device);
  EXPECT_EQ(1, t3.dim());
  EXPECT_EQ(4, t3.sizes()[0]);
  EXPECT_EQ(1.f, t3[2].item().to<float>());
  EXPECT_EQ(0.f, t3[0].item().to<float>());

  t::Tensor t4 = encoder.encode_action(g::Action::RT, device);
  EXPECT_EQ(1, t4.dim());
  EXPECT_EQ(4, t4.sizes()[0]);
  EXPECT_EQ(1.f, t4[3].item().to<float>());
  EXPECT_EQ(0.f, t4[0].item().to<float>());
}

TEST(TestGridActionEncoder, TestDecode1){
  m::GridEnv env(4, m::GridEnvMode::StaticSimple);
  m::GridActionEncoder encoder(env);

  float data[] = {0.1, 0.1, 0.7, 0.1};
  t::Tensor t = t::from_blob(data, {4});
  g::Action action = encoder.decode_action(t);
  EXPECT_EQ(g::Action::LF, action);

  data[0] = 0.9; data[2] = 0.01;
  t = t::from_blob(data, {4});
  action = encoder.decode_action(t);
  EXPECT_EQ(g::Action::UP, action);

  data[1] = 0.95; data[0] = 0.0001;
  t = t::from_blob(data, {4});
  action = encoder.decode_action(t);
  EXPECT_EQ(g::Action::DN, action);

  data[3] = 0.5; data[1] = 0.49;
  t  = t::from_blob(data, {4});
  action = encoder.decode_action(t);
  EXPECT_EQ(g::Action::RT, action);
}
