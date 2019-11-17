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

    t::Tensor tstate = model.state_encoder.encode_state(env.get_state(ins));
    t::Tensor tstate_dev = tstate.to(device);
    t::Tensor taction_dev = model.model->forward(tstate_dev);
    t::Tensor taction = taction_dev.to(cpu_device);
    g::Action action = model.action_encoder.decode_action(taction);
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
