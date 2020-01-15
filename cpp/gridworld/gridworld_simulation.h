#ifndef GRIDWORLD_SIMULATION
#define GRIDWORLD_SIMULATION

#include <gridworld.h>
#include <torch/torch.h>

namespace gridworld_pt {

namespace g = gridworld;
namespace t = torch;

template <typename Model>
int simulate_gridworld(GridEnv& env, Model& model, uint max_steps, t::Device device, bool display = false){
  t::Device cpu_device(t::kCPU);

  g::GridWorld ins = env.create();
  uint step_count = 0;
  while (not env.is_termination(ins) && step_count < max_steps){
    if (display)
      env.display(ins);

    t::Tensor tstate_dev = model.state_encoder.encode_state(env.get_state(ins), device);
    t::Tensor taction_dev = model.model->forward(tstate_dev);
    g::Action action = model.action_encoder.decode_action(taction_dev);
    env.apply_action(ins, action);

    if (display){
      switch (action){
      case g::Action::UP: s::cout << "Up" << s::endl; break;
      case g::Action::DN: s::cout << "Down" << s::endl; break;
      case g::Action::LF: s::cout << "Left" << s::endl; break;
      case g::Action::RT: s::cout << "Right" << s::endl; break;
      default: assert(false);
      }
    }

    step_count++;
  }

  if (display)
    env.display(ins);

  return env.get_reward(ins);
}

//model simulation specifically for distributional qlearning
template <typename Model>
int simulate_gridworld_dq(GridEnv& env, Model& model, uint datoms, uint max_steps, t::Device device, bool display = false){
  t::Device cpu_device(t::kCPU);

  g::GridWorld ins = env.create();

  s::vector<float> support_vec(datoms);
  float zdelta = (env.max_reward(ins) - env.min_reward(ins)) / (float)datoms;
  for (uint i = 0; i < support_vec.size(); ++i)
    support_vec[i] = env.min_reward(ins) + i * zdelta;
  t::Tensor support = t::from_blob(support_vec.data(), {datoms});
  t::Tensor support_dev = support.to(device);

  uint step_count = 0;
  while (not env.is_termination(ins) && step_count < max_steps){
    if (display)
      env.display(ins);

    t::Tensor tstate_dev = model.state_encoder.encode_state(env.get_state(ins), device);
    t::Tensor taction_dev = model.model->forward(tstate_dev);
    taction_dev = (taction_dev * support_dev).sum(-1);
    g::Action action = model.action_encoder.decode_action(taction_dev);
    env.apply_action(ins, action);

    if (display){
      switch (action){
      case g::Action::UP: s::cout << "Up" << s::endl; break;
      case g::Action::DN: s::cout << "Down" << s::endl; break;
      case g::Action::LF: s::cout << "Left" << s::endl; break;
      case g::Action::RT: s::cout << "Right" << s::endl; break;
      default: assert(false);
      }
    }

    step_count++;
  }

  if (display)
    env.display(ins);

  return env.get_reward(ins);
}

// model simulation specifically for actor critic models
template <typename Model>
int simulate_gridworld_ac(GridEnv& env, Model& model, uint max_steps, t::Device device, bool display = false){
  t::Device cpu_device(t::kCPU);

  g::GridWorld ins = env.create();
  uint step_count = 0;
  while (not env.is_termination(ins) && step_count < max_steps){
    if (display)
      env.display(ins);

    t::Tensor tstate_dev = model.state_encoder.encode_state(env.get_state(ins), device);
    t::Tensor taction_dev = model.model->actor_forward(tstate_dev);
    g::Action action = model.action_encoder.decode_action(taction_dev);
    env.apply_action(ins, action);

    if (display){
      switch (action){
      case g::Action::UP: s::cout << "Up" << s::endl; break;
      case g::Action::DN: s::cout << "Down" << s::endl; break;
      case g::Action::LF: s::cout << "Left" << s::endl; break;
      case g::Action::RT: s::cout << "Right" << s::endl; break;
      default: assert(false);
      }
    }

    step_count++;
  }

  if (display)
    env.display(ins);

  return env.get_reward(ins);
}


} // gridworld_pt

#endif//GRIDWORLD_SIMULATION
