#include <cassert>
#include <ctime>
#include <cctype>
#include <string>
#include <random>
#include <chrono>
#include <iostream>
#include <sstream>
#include <filesystem>

#include <type_alias.h>
#include <types.h>
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

#include <boost/program_options.hpp>

#include <Wt/WApplication.h>
#include <Wt/WServer.h>
#include <Wt/WContainerWidget.h>
#include <Wt/WText.h>
#include <Wt/WPushButton.h>
#include <Wt/WLineEdit.h>

namespace s = std;
namespace t = torch;
namespace R = rlgames;
namespace w = Wt;
namespace po= boost::program_options;

constexpr ubyte SZ = 9;
constexpr uint action_size = R::ZeroGoActionEncoder<SZ>::action_size();
using ModelContainerType = R::ModelContainer<R::ZeroModelSmall, R::ZeroGoStateEncoder<SZ>, R::ZeroGoActionEncoder<SZ>, t::optim::Adam>;
using AgentType = R::ZeroAgent<ModelContainerType, R::null_distribution<action_size>, R::Splitmix, R::GoBoard<SZ>, R::GoGameState<SZ>, action_size>;

s::string model_config;
s::string model_params;
s::string optimizer_params;

s::string board_to_html(s::string board_str){
  s::string ret("<pre>");
  for (uint i = 0; i < board_str.length(); ++i){
    if (board_str[i] != '\n')
      ret += board_str[i];
    else
      ret += "</pre><pre>";
  }
  ret += "</pre>";
  return ret;
}

class GoGameWidget : public w::WContainerWidget {
public:
  GoGameWidget():
    mGoGameState(),
    mDevice(t::Device(t::kCPU)),
    mModelContainer(
      R::ZeroModelSmall(R::ZeroGoStateEncoder<SZ>().state_size(), action_size, R::load_model_option<R::ZeroModelSmallOptions>(model_config)),
      R::ZeroGoStateEncoder<SZ>(),
      R::ZeroGoActionEncoder<SZ>(),
      1E-5F),
    mAgent(mModelContainer, mDevice, 3200, 0, 0., 0., rand()){

    // setup widget
    mBoard = addWidget(s::make_unique<w::WText>(""));
    addWidget(s::make_unique<w::WBreak>());
    mUserInput = addWidget(s::make_unique<w::WLineEdit>());
    mUserInput->setMaxLength(3);
    mPlayButton = addWidget(s::make_unique<w::WPushButton>("play"));
    mResetButton = addWidget(s::make_unique<w::WPushButton>("restart"));

    mPlayButton->clicked().connect(this, &GoGameWidget::play);
    mResetButton->clicked().connect(this, &GoGameWidget::reset);

    display_board();

    R::load_model(mModelContainer, model_params, optimizer_params, mDevice);
    mModelContainer.model->to(mDevice);
  }
private:
  R::GoGameState<SZ> mGoGameState;
  t::Device          mDevice;
  ModelContainerType mModelContainer;
  AgentType          mAgent;

  w::WText*       mBoard;
  w::WLineEdit*   mUserInput;
  w::WPushButton* mPlayButton;
  w::WPushButton* mResetButton;

  void play(){
    s::string input_str = mUserInput->valueText().toUTF8();
    mUserInput->setValueText("");

    if (mGoGameState.is_over())
      return;

    R::Move move = R::string_to_move(input_str, SZ);
    if (move.mty == R::M::Unknown){
      //TODO: log error parsing input
      return;
    }

    mGoGameState.apply_move(move);

    display_board();

    R::Move agent_move = mAgent.select_move(mGoGameState);

    mGoGameState.apply_move(agent_move);

    display_board();
  }

  void reset(){
    mGoGameState = R::GoGameState<SZ>();
    display_board();
  }

  void display_board(){
    s::stringstream ss;
    mGoGameState.board().print(ss);
    s::string s = ss.str();
    mBoard->setText(w::WString(board_to_html(s)));
  }
};

s::unique_ptr<w::WApplication> createApplication(const w::WEnvironment& env){
  auto app = w::cpp14::make_unique<w::WApplication>(env);

  app->setTitle("Go");

  app->root()->addWidget(w::cpp14::make_unique<GoGameWidget>());
  app->useStyleSheet("css/go.css");
  return app;
}

int main(int argc, char* argv[]){
  po::options_description desc("Parameters");
  try {
    srand(time(nullptr));

    desc.add_options()
      ("model_config", po::value<s::string>()->required(), "model hyperparameter configuration file")
      ("model_params", po::value<s::string>()->required(), "model parameters file")
      ("optimizer_params", po::value<s::string>()->required(), "optimizer parameters file")
      ("docroot", po::value<s::string>()->required(), "root to the document path")
      ("http-address", po::value<s::string>()->required(), "http server address")
      ("http-port", po::value<int>()->required(), "http server port");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    model_config = vm["model_config"].as<s::string>();
    model_params = vm["model_params"].as<s::string>();
    optimizer_params = vm["optimizer_params"].as<s::string>();

    w::WServer server(argc, argv, WTHTTP_CONFIGURATION);
    server.addEntryPoint(w::EntryPointType::Application, createApplication);

    server.run();
  } catch (po::error& e){
    s::cerr << e.what() << s::endl;
    s::cerr << desc << s::endl;
  } catch (w::WServer::Exception& e){
    s::cerr << e.what() << s::endl;
  } catch (s::exception& e){
    s::cerr << "exception: " << e.what() << s::endl;
  }
}
