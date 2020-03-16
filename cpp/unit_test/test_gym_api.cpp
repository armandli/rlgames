#include <iostream>

#include <gym/gym.h>

#include <string>
#include <rapidjson/document.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

namespace s = std;
namespace g = gym;
namespace j = rapidjson;

struct TestGymClient : ::testing::Test {
  TestGymClient(): client("127.0.0.1", 5000) {}
  ~TestGymClient(){}

  g::GymClient client;
};

TEST_F(TestGymClient, TestPost1){
  g::ServerResponse resp = client.post("/v1/envs/", "{\"env_id\":\"CartPole-v0\"}");
  ASSERT_EQ(200, resp.code);
}

TEST_F(TestGymClient, TestGet1){
  g::ServerResponse resp1 = client.post("/v1/envs/", "{\"env_id\":\"CartPole-v0\"}");
  EXPECT_EQ(200, resp1.code);

  j::Document doc;
  doc.Parse(resp1.msg.c_str());
  s::string instance_id = doc["instance_id"].GetString();

  s::string req = "/v1/envs/" + instance_id + "/action_space/";
  g::ServerResponse resp2 = client.get(req);
  ASSERT_EQ(200, resp2.code);
}

struct MockGymEnvironment : g::GymEnvironment {
  using g::GymEnvironment::parse_space;
  using g::GymEnvironment::get_state_space;
  using g::GymEnvironment::get_action_space;
  using g::GymEnvironment::parse_reset_response;
  using g::GymEnvironment::build_monitor_command;
  using g::GymEnvironment::start_monitor;
  using g::GymEnvironment::stop_monitor;
  using g::GymEnvironment::reset_instance;
  using g::GymEnvironment::action_to_command;
  using g::GymEnvironment::parse_step_response;
  using g::GymEnvironment::make_step;

  MockGymEnvironment(const s::string& name, const s::string& addr, uint port): g::GymEnvironment(name, addr, port) {}
};

struct TestGymEnvironment : ::testing::Test {
  TestGymEnvironment(): env("CartPole-v0", "0.0.0.0", 5000) {}
  ~TestGymEnvironment(){}

  MockGymEnvironment env;
};

TEST_F(TestGymEnvironment, TestParseSpace1){
  s::string msg = "{\"info\":{\"n\":4,\"name\":\"Discrete\"}}";
  g::Space s = env.parse_space(msg);
  EXPECT_EQ(g::ST::DISCRETE, s.type);
  EXPECT_EQ(4, s.shape[0]);
}

TEST_F(TestGymEnvironment, TestParseSpace2){
  s::string msg = "{\"info\":{\"high\":[4.800000190734863,3.4028234663852886e+38,0.41887903213500977,3.4028234663852886e+38],\"low\":[-4.800000190734863,-3.4028234663852886e+38,-0.41887903213500977,-3.4028234663852886e+38],\"name\":\"Box\",\"shape\":[4]}}";
  g::Space s = env.parse_space(msg);
  EXPECT_EQ(g::ST::BOX, s.type);
  EXPECT_EQ(4, s.shape[0]);
  EXPECT_EQ(4, s.high.size());
  EXPECT_FLOAT_EQ(4.800000190734863, s.high[0]);
  EXPECT_FLOAT_EQ(3.4028234663852886e+38, s.high[1]);
  EXPECT_FLOAT_EQ(3.4028234663852886e+38, s.high[3]);
  EXPECT_FLOAT_EQ(-4.800000190734863, s.low[0]);
}

TEST_F(TestGymEnvironment, TestCreate1){
  g::GymInstance ins = env.create();
  s::string id = ins.instance_id();
  g::Space as = ins.action_space();
  g::Space ss = ins.state_space();
  EXPECT_TRUE(id.size() > 0);
  EXPECT_EQ(g::ST::DISCRETE, as.type);
  EXPECT_EQ(g::ST::BOX, ss.type);
  EXPECT_EQ(2, as.shape[0]);
}

TEST_F(TestGymEnvironment, TestParseResetResponse1){
  g::GymInstance ins = env.create();
  s::string resp = "{\"observation\":[0.006926668491562114,0.011587148392201894,-0.01376908822847809,-0.03993594806945655]}";
  g::State st = env.parse_reset_response(ins, resp);
  EXPECT_EQ(4, st.size());
  EXPECT_FLOAT_EQ(0.0069266684915621143, st[0]);
}

TEST_F(TestGymEnvironment, TestParseResetResponse2){
  g::GymInstance ins = env.create();
  g::Space discrete;
  discrete.type = g::ST::DISCRETE;
  discrete.shape.push_back(3);
  ins.set_state_space(discrete);

  s::string resp = "{\"observation\":2}";
  g::State st = env.parse_reset_response(ins, resp);
  EXPECT_EQ(1, st.size());
  EXPECT_EQ(2, st[0]);
}

TEST_F(TestGymEnvironment, TestParseResetResponse3){
  g::GymInstance ins = env.create();
  g::Space box3d;
  box3d.type = g::ST::BOX;
  box3d.shape.push_back(2);
  box3d.shape.push_back(160);
  box3d.shape.push_back(3);
  uint flatten_size = 2 * 160 * 3;
  for (uint i = 0; i < flatten_size; ++i){
    box3d.low.push_back(0);
    box3d.high.push_back(255);
  }
  ins.set_state_space(box3d);

  s::string resp = "{\"observation\":[[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],[[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111],[228,111,111]]]}";

  g::State st = env.parse_reset_response(ins, resp);
  EXPECT_EQ(flatten_size, st.size());

  for (uint i = 0; i < st.size(); ++i){
    EXPECT_TRUE(st[i] >= 0.F);
    EXPECT_TRUE(st[i] <= 255.F);
  }
}

TEST_F(TestGymEnvironment, TestParseStepResponse1){
  g::GymInstance ins = env.create();
  s::string msg = "{\"done\":false,\"info\":{},\"observation\":[0.007158411459406152,0.20690381278981043,-0.014567807189867221,-0.3369311568917657],\"reward\":1.0}";
  env.parse_step_response(ins, msg);
  EXPECT_FALSE(ins.is_termination());
  EXPECT_FALSE(env.is_termination(ins));
  EXPECT_EQ(4, ins.state().size());
  EXPECT_FLOAT_EQ(-0.3369311568917657, ins.state()[3]);
  EXPECT_FLOAT_EQ(1.0, ins.reward());
  EXPECT_FLOAT_EQ(1.0, env.get_reward(ins));
}

TEST_F(TestGymEnvironment, TestParseStepResponse2){
  g::GymInstance ins = env.create();
  s::string msg = "{\"done\":true,\"info\":{},\"observation\":[0.007158411459406152,0.20690381278981043,-0.014567807189867221,-0.3369311568917657],\"reward\":2.0}";
  env.parse_step_response(ins, msg);
  EXPECT_TRUE(ins.is_termination());
  EXPECT_EQ(4, ins.state().size());
  EXPECT_FLOAT_EQ(-0.3369311568917657, ins.state()[3]);
  EXPECT_FLOAT_EQ(2.0, ins.reward());
  EXPECT_FLOAT_EQ(2.0, env.get_reward(ins));
}

TEST_F(TestGymEnvironment, TestActionToCommand1){
  g::GymInstance ins = env.create();
  g::Action action; action.push_back(1);
  s::string cmd = env.action_to_command(ins, action);
  EXPECT_TRUE(cmd == "{\"action\":1}");
}
