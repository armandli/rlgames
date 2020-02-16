#ifndef RLGAMES_MODEL_BASE
#define RLGAMES_MODEL_BASE

#include <cassert>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <fstream>

#include <pytorch_util.h>
#include <encoders/go_zero_encoder.h>
#include <experience/zero_episodic_buffer.h>

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>

#include <torch/torch.h>

namespace rlgames {

namespace s = std;
namespace j = rapidjson;
namespace t = torch;

template <typename Option>
Option load_model_option(const s::string& filename){
  s::ifstream ifs(filename);
  j::IStreamWrapper isw(ifs);

  j::Document doc;
  doc.ParseStream(isw);
  assert(doc.IsArray());

  s::vector<TensorDim> dims;

  for (auto& layer : doc.GetArray()){
    assert(layer.IsObject());
    if (layer.HasMember("tensordim")){
      auto& dim = layer["tensordim"];
      assert(dim.IsObject());
      TensorDim dimobj;
      for (auto& dimension : dim.GetObject()){
        if (s::strcmp(dimension.name.GetString(), "i") == 0){
          assert(dimension.value.IsInt());
          dimobj.i = dimension.value.GetInt();
        } else if (s::strcmp(dimension.name.GetString(), "j") == 0){
          assert(dimension.value.IsInt());
          dimobj.j = dimension.value.GetInt();
        } else if (s::strcmp(dimension.name.GetString(), "k") == 0){
          assert(dimension.value.IsInt());
          dimobj.k = dimension.value.GetInt();
        }
      }
      dims.push_back(dimobj);
    }
  }
  return Option(dims);
}

template <typename NNModel, typename SE, typename AE, typename Optim>
struct ModelContainer {
  NNModel model;
  SE      state_encoder;
  AE      action_encoder;
  Optim   optimizer;

  ModelContainer(NNModel&& m, SE&& se, AE&& ae, float learning_rate):
    model(s::move(m)),
    state_encoder(s::move(se)),
    action_encoder(s::move(ae)),
    optimizer(m->parameters(), t::optim::AdamOptions(learning_rate))
  {}
};

template <typename Model>
void save_model(Model& model, const s::string& model_file, const s::string& optimizer_file){
  t::save(model.model, model_file);
  t::save(model.optimizer, optimizer_file);
}

template <typename Model>
void load_model(Model& model, const s::string& model_file, const s::string& optimizer_file){
  t::load(model.model, model_file);
  t::load(model.optimizer, optimizer_file);
}

//TODO: is there anyway we can record and monitor the loss ?
template <typename Model>
void train(Model& model, ZeroExperience& exp){
  model.model->zero_grad();

  t::Tensor visit_sums = t::sum(exp.visit_counts, -1);
  t::Tensor visit_counts = exp.visit_counts / visit_sums;

  TensorP avout = model.model->forward(TensorP(exp.boards, exp.states));

  t::Tensor policy_loss = t::mean(-1.F * visit_counts.detach() * t::log(avout.x));
  t::Tensor value_loss = t::mse_loss(avout.y, exp.rewards.detach());
  t::Tensor loss = policy_loss + value_loss;

  loss.backward();
  model.optimizer.step();
}

} // rlgames

#endif//RLGAMES_MODEL_BASE
