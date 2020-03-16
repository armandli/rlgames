#ifndef RLGAMES_GYM
#define RLGAMES_GYM

//modified source code from gym_http_api c++ version

#include <cassert>
#include <string>
#include <vector>
#include <random>
#include <memory>

#include <type_alias.h>

#include <curl/curl.h>

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

namespace gym {

namespace s = std;
namespace j = rapidjson;

uint curl_save_to_string(void* buffer, uint size, uint nmemb, void* userp){
  s::string* str = static_cast<s::string*>(userp);
  const uint bytes = nmemb * size;
  str->append(static_cast<char*>(buffer), bytes);
  return bytes;
}

struct ServerResponse {
  uint code;
  s::string msg;

  ServerResponse() = default;
  ServerResponse(uint code, const s::string& msg): code(code), msg(msg) {}
};

class GymClient {
  s::shared_ptr<CURL>       c;
  s::shared_ptr<curl_slist> headers;
  s::vector<char>           error_buf;
  s::string                 address;
  sint                      port;
public:
  GymClient(const s::string& addr, sint port):
    address(addr), port(port) {

    CURL* c = curl_easy_init();
		curl_easy_setopt(c, CURLOPT_NOSIGNAL, 1);
		curl_easy_setopt(c, CURLOPT_CONNECTTIMEOUT_MS, 3000);
		curl_easy_setopt(c, CURLOPT_IPRESOLVE, CURL_IPRESOLVE_V4);
		curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, true);
		curl_easy_setopt(c, CURLOPT_SSL_VERIFYPEER, 0);
		curl_easy_setopt(c, CURLOPT_SSL_VERIFYHOST, 0);
		curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, &curl_save_to_string);
		error_buf.assign(CURL_ERROR_SIZE, 0);
		curl_easy_setopt(c, CURLOPT_ERRORBUFFER, error_buf.data());
		this->c.reset(c, s::ptr_fun(curl_easy_cleanup));
		headers.reset(curl_slist_append(0, "Content-Type: application/json"), s::ptr_fun(curl_slist_free_all));
  }

  ServerResponse get(const s::string& route){
    s::string url = "http://" + address + route;
    ServerResponse response;

    curl_easy_setopt(c.get(), CURLOPT_URL, url.c_str());
    curl_easy_setopt(c.get(), CURLOPT_PORT, port);
    curl_easy_setopt(c.get(), CURLOPT_WRITEDATA, &response.msg);
    curl_easy_setopt(c.get(), CURLOPT_POST, 0);
    curl_easy_setopt(c.get(), CURLOPT_HTTPHEADER, 0);

    CURLcode r = curl_easy_perform(c.get());
    r = curl_easy_getinfo(c.get(), CURLINFO_RESPONSE_CODE, &response.code);
    return response;
  }

  ServerResponse post(const s::string& route, const s::string& post_data){
    s::string url = "http://" + address + route;
    ServerResponse response;

    curl_easy_setopt(c.get(), CURLOPT_URL, url.c_str());
    curl_easy_setopt(c.get(), CURLOPT_PORT, port);
    curl_easy_setopt(c.get(), CURLOPT_WRITEDATA, &response.msg);
    curl_easy_setopt(c.get(), CURLOPT_POST, 1);
    curl_easy_setopt(c.get(), CURLOPT_POSTFIELDS, post_data.c_str());
    curl_easy_setopt(c.get(), CURLOPT_POSTFIELDSIZE_LARGE, (curl_off_t)post_data.size());
    curl_easy_setopt(c.get(), CURLOPT_HTTPHEADER, headers.get());

    CURLcode r = curl_easy_perform(c.get());
    r = curl_easy_getinfo(c.get(), CURLINFO_RESPONSE_CODE, &response.code);
    return response;
  }
};

//NOTE: there is also Tuple type, but the http service has issue handling it
enum class ST {
  DISCRETE,
  BOX,
  UNKNOWN,
};

struct Space {
  s::vector<uint>  shape;
  s::vector<float> high;
  s::vector<float> low;
  ST               type;
};

template <typename RENG>
s::vector<float> sample_space(const Space& space, RENG& reng){
  s::vector<float> ret;
  switch (space.type){
  case ST::DISCRETE: {
    s::uniform_int_distribution<int> dist(0, space.shape[0] - 1);
    ret.push_back(dist(reng));
  }
  break;
  case ST::BOX: {
    s::uniform_real_distribution<float> dist(0.F, 1.F);
    uint sz = 1;
    for (uint dim : space.shape)
      sz *= dim;
    assert(space.high.size() == sz);
    assert(space.low.size() == sz);
    for (uint c = 0; c < sz; ++c)
      ret.push_back((space.high[c] - space.low[c]) * dist(reng) + space.low[c]);
  }
  break;
  default:
    assert(false);
  }
  return ret;
}

//TODO: maybe optimize for size = 1 discrete case
using State = s::vector<float>;
using Action = s::vector<float>;

constexpr char GYM_ROUTE[] = "/v1/envs/";

class GymInstance {
  s::string mInstanceId;
  Space     mStateSpace;
  Space     mActionSpace;
  State     mState;
  float     mReward;
  bool      mIsReset;
  bool      mIsTermination;
  bool      mMonitorStarted;
public:
  GymInstance(const s::string& id, const Space& ss, const Space& as): mInstanceId(id), mStateSpace(ss), mActionSpace(as), mReward(0.F), mIsReset(false), mIsTermination(false), mMonitorStarted(false) {}
  ~GymInstance(){
    //should stop monitoring
    assert(mMonitorStarted == false);
  }

  const s::string& instance_id() const {
    return mInstanceId;
  }
  const Space& state_space() const {
    return mStateSpace;
  }
  const Space& action_space() const {
    return mActionSpace;
  }
  const State& state() const {
    return mState;
  }
  float reward() const {
    return mReward;
  }
  bool is_reset() const {
    return mIsReset;
  }
  bool is_termination() const {
    return mIsTermination;
  }
  bool is_monitor_started() const {
    return mMonitorStarted;
  }
  
  void reset(){ mIsReset = true; }
  void set_monitor(bool val){ mMonitorStarted = val; }
  void set_state(const State& s){ mState = s; }
  void set_reward(float r){ mReward = r; }
  void set_termination(bool t){ mIsTermination = t; }
  void set_state_space(const Space& s){ mStateSpace = s; }
  void set_action_space(const Space& s){ mActionSpace = s; }
};

class GymEnvironment {
  GymClient mClient;
  s::string mEnvName;
  s::string mMonitorFilePath;
  bool      mShouldMonitor;
protected:
  struct NestedArrayParser {
    template <typename T>
    void operator()(State& ret, T& layer, uint level){
      if (level == 0){
        assert(layer.IsNumber());
        ret.push_back(layer.GetFloat());
        return;
      }

      assert(layer.IsArray());
      for (auto& v : layer.GetArray())
        (*this)(ret, v, level - 1);
    }
  };

  Space parse_space(const s::string& msg){
    Space space;

    j::Document doc;
    doc.Parse(msg.c_str());
    assert(doc.IsObject());
    assert(doc.HasMember("info"));
    auto info = doc["info"].GetObject();
    assert(info.HasMember("name"));
    s::string space_type = info["name"].GetString();
    if (space_type == "Discrete"){
      assert(info.HasMember("n"));
      space.type = ST::DISCRETE;
      uint dim = info["n"].GetInt();
      space.shape.push_back(dim);
    } else if (space_type == "Box"){
      assert(info.HasMember("shape"));
      assert(info.HasMember("high"));
      assert(info.HasMember("low"));
      auto& shape = info["shape"];
      auto& highs = info["high"];
      auto& lows  = info["low"];
      assert(shape.IsArray());
      assert(highs.IsArray());
      assert(lows.IsArray());

      space.type = ST::BOX;
      for (auto& s : shape.GetArray()){
        assert(s.IsInt());
        space.shape.push_back(s.GetInt());
      }
      for (auto& h : highs.GetArray()){
        assert(h.IsNumber());
        space.high.push_back(h.GetFloat());
      }
      for (auto& l : lows.GetArray()){
        assert(l.IsNumber());
        space.low.push_back(l.GetFloat());
      }
    } else {
      assert(!!!"unknown space");
    }
    return space;
  }

  Space get_state_space(const s::string& ins_id){
    s::string req = GYM_ROUTE + ins_id + "/observation_space/";
    ServerResponse resp = mClient.get(req);
    assert(resp.code == 200);
    return parse_space(resp.msg);
  }

  Space get_action_space(const s::string& ins_id){
    s::string req = GYM_ROUTE + ins_id + "/action_space/";
    ServerResponse resp = mClient.get(req);
    assert(resp.code == 200);
    return parse_space(resp.msg);
  }

  State parse_observation(GymInstance& ins, const s::string& resp){
    State ret;

    ST stype = ins.state_space().type;
    assert(stype != ST::UNKNOWN);

    switch (stype){
    case ST::DISCRETE:
      ret.reserve(1);
      break;
    case ST::BOX:
      ret.reserve(ins.state_space().high.size());
      break;
    default: assert(false);
    }

    NestedArrayParser layer_parser;

    j::Document doc;
    doc.Parse(resp.c_str());
    assert(doc.IsObject());
    assert(doc.HasMember("observation"));
    switch (stype){
    case ST::DISCRETE:
      assert(doc["observation"].IsNumber());

      ret.push_back(doc["observation"].GetInt());
      break;
    case ST::BOX:
      assert(doc["observation"].IsArray());

      layer_parser(ret, doc["observation"], ins.state_space().shape.size());
      break;
    default: assert(false);
    }

    return ret;
  }

  State parse_reset_response(GymInstance& ins, const s::string& resp){
    return parse_observation(ins, resp);
  }

  s::string build_monitor_command(GymInstance& ins){
    j::Document doc;
    doc.SetObject();
    j::Document::AllocatorType& allocator = doc.GetAllocator();

    j::Value instance_key("instance_id");
    j::Value instance_val; instance_val.SetString(ins.instance_id().c_str(), allocator);
    doc.AddMember(instance_key, instance_val, allocator);

    j::Value directory_key("directory");
    j::Value directory_val; directory_val.SetString(mMonitorFilePath.c_str(), allocator);
    doc.AddMember(directory_key, directory_val, allocator);

    j::StringBuffer buffer;
    j::Writer<j::StringBuffer> writer(buffer);
    doc.Accept(writer);
    const char* output = buffer.GetString();
    return output;
  }

  void start_monitor(GymInstance& ins){
    s::string req = GYM_ROUTE + ins.instance_id() + "/monitor/start/";
    s::string cmd = build_monitor_command(ins);
    ServerResponse resp = mClient.post(req, cmd);
    assert(resp.code == 200);
  }

  void stop_monitor(GymInstance& ins){
    if (not ins.is_monitor_started()) return;

    s::string req = GYM_ROUTE + ins.instance_id() + "monitor/stop";
    ServerResponse resp = mClient.post(req, "");
    assert(resp.code == 200);
  }

  void reset_instance(GymInstance& ins){
    s::string req = GYM_ROUTE + ins.instance_id() + "/reset/";
    s::string cmd = "{\"instance_id:\"" + ins.instance_id() + "\"}";
    ServerResponse resp = mClient.post(req, cmd);
    assert(resp.code == 200);
    ins.reset();
    ins.set_state(parse_reset_response(ins, resp.msg));

    if (mShouldMonitor){
      start_monitor(ins);
      ins.set_monitor(true);
    }
  }

  s::string action_to_command(GymInstance& ins, const Action& action){
    j::Document doc;
    doc.SetObject();
    j::Document::AllocatorType& allocator = doc.GetAllocator();

    j::Value action_key("action");
    if (ins.action_space().type == ST::DISCRETE){
      j::Value action_value((int)action[0]);
      doc.AddMember(action_key, action_value, allocator);
    } else {
      j::Value action_values(j::kArrayType);
      for (uint i = 0; i < action.size(); ++i)
        action_values.PushBack(j::Value().SetFloat(action[i]), allocator);
      doc.AddMember(action_key, action_values, allocator);
    }
    j::StringBuffer buffer;
    j::Writer<j::StringBuffer> writer(buffer);
    doc.Accept(writer);
    const char* output = buffer.GetString();
    return output;
  }

  void parse_step_response(GymInstance& ins, const s::string& msg){
    State state;

    j::Document doc;
    doc.Parse(msg.c_str());
    assert(doc.IsObject());
    assert(doc.HasMember("observation"));
    assert(doc.HasMember("reward"));
    assert(doc.HasMember("done"));
    assert(doc.HasMember("info"));
    bool done = doc["done"].GetBool();
    float reward = doc["reward"].GetFloat();

    State st = parse_observation(ins, msg);
    ins.set_state(st);
    ins.set_termination(done);
    ins.set_reward(reward);
    //TODO: what to do with info ?
  }

  void make_step(GymInstance& ins, const Action& action){
    s::string req = GYM_ROUTE + ins.instance_id() + "/step/";
    s::string cmd = action_to_command(ins, action);
    ServerResponse resp = mClient.post(req, cmd);
    assert(resp.code == 200);
    parse_step_response(ins, resp.msg);
  }
public:
  GymEnvironment(const s::string& name, const s::string& address, sint port, bool monitor = false, const s::string& monitor_filepath = ""):
    mClient(address, port),
    mEnvName(name),
    mMonitorFilePath(monitor_filepath),
    mShouldMonitor(monitor)
  {}

  Space action_space(const GymInstance& ins){
    return ins.action_space();
  }
  Space state_space(const GymInstance& ins){
    return ins.state_space();
  }

  GymInstance create(){
    s::string env_cmd = "{\"env_id\":\"" + mEnvName + "\"}";
    ServerResponse resp = mClient.post(GYM_ROUTE, env_cmd);
    assert(resp.code == 200);

    j::Document doc;
    doc.Parse(resp.msg.c_str());
    assert(doc.IsObject());
    assert(doc.HasMember("instance_id"));

    s::string ins_id = doc["instance_id"].GetString();
    Space state_space = get_state_space(ins_id);
    Space action_space = get_action_space(ins_id);
    return GymInstance(ins_id, state_space, action_space);
  }
  bool is_termination(GymInstance& ins){
    return ins.is_termination();
  }
  void terminate(GymInstance& ins){
    stop_monitor(ins);
  }
  void apply_action(GymInstance& ins, const Action& action){
    if (not ins.is_reset())
      reset_instance(ins);
    make_step(ins, action);
  }
  State get_state(GymInstance& ins){
    if (not ins.is_reset())
      reset_instance(ins);
    return ins.state();
  }
  float get_reward(GymInstance& ins){
    return ins.reward();
  }
};

} // gym

#endif//RLGAMES_GYM
