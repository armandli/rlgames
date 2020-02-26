#include <cassert>
#include <ctime>
#include <cctype>
#include <string>
#include <random>
#include <chrono>
#include <iostream>
#include <filesystem>

#include <type_alias.h>
#include <types.h>
#include <input_parser.h>
#include <go_types.h>
#include <splitmix.h>
#include <null_distribution.h>
#include <pytorch_util.h>
#include <encoders/go_action_encoder.h>
#include <encoders/go_zero_encoder.h>
#include <models/model_base.h>
#include <models/zero_model_small.h>
#include <agents/zero_agent.h>

#include <torch/torch.h>

namespace s = std;
namespace c = s::chrono;
namespace t = torch;
namespace R = rlgames;

constexpr ubyte SZ= 9;

int main(int argc, const char* argv[]){
  s::string model_config_file;
  s::string model_file;
  s::string optimizer_file;
  if (argc != 4){
    s::cout << "Usage: human_vs_zero_go <model_config_file> <model_file> <optimizer_file>" << s::endl;
    s::exit(1);
  }

  model_config_file = argv[1];
  model_file = argv[2];
  optimizer_file = argv[3];

  if (not s::filesystem::exists(model_config_file)){
    s::cout << "model configuration file does not exist" << s::endl;
    s::exit(1);
  }

  if (not s::filesystem::exists(model_file) || not s::filesystem::exists(optimizer_file)){
    s::cout << "model file or optimizer file does not exist" << s::endl;
    s::exit(1);
  }

  srand(time(nullptr));

  t::Device device(t::kCPU);
  if (t::cuda::is_available()){
    device = t::Device(t::kCUDA);
  }

  R::ZeroGoStateEncoder<SZ> state_encoder;
  R::ZeroGoActionEncoder<SZ> action_encoder;

  R::TensorDimP state_size = state_encoder.state_size();
  constexpr uint action_size = R::ZeroGoActionEncoder<SZ>::action_size();

  R::ModelContainer<R::ZeroModelSmall, R::ZeroGoStateEncoder<SZ>, R::ZeroGoActionEncoder<SZ>, t::optim::Adam> model_container(
    R::ZeroModelSmall(state_size, action_size, R::load_model_option<R::ZeroModelSmallOptions>(model_config_file)),
    s::move(state_encoder),
    s::move(action_encoder),
    1E-5F //learning_rate
  );

  R::load_model(model_container, model_file, optimizer_file, device);

  model_container.model->to(device);

  R::ZeroAgent<decltype(model_container), R::null_distribution<action_size>, R::Splitmix, R::GoBoard<SZ>, R::GoGameState<SZ>, action_size> agent(
    model_container,
    device,
    3200,   /*max expansion*/
    0.2F,   /*exploration factor*/
    1000.,  /*dirichlet distribution alpha*/
    0.,     /*noise factor*/
    rand()  /*random seed*/
  );

  R::GoGameState<SZ> state;
  R::Player turn = R::Player::Black;

  while (not state.is_over()){
    s::cout << state.board() << s::endl;

    R::Move move(R::M::Pass);

    switch (turn){
    case R::Player::Black: move = R::parse_move(SZ); break;
    case R::Player::White: move = agent.select_move(state); break;
    default: assert(false);
    }

    state.apply_move(move);
    turn = R::other_player(turn);
  }
  s::cout << state.board() << s::endl;

  R::Player winner = state.winner();
  switch (winner){
  case R::Player::Black: case R::Player::White:
    s::cout << winner << " won";
    break;
  case R::Player::Unknown:
    s::cout << "ties";
    break;
  default: assert(false);
  }
  s::cout << s::endl;
}
