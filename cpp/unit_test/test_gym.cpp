#include <vector>
#include <string>
#include <memory>
#include <iostream>

#include <type_alias.h>

#include <curl/curl.h>

#include <rapidjson/document.h>

namespace s = std;
namespace j = rapidjson;

uint curl_save_to_string(void* buffer, uint size, uint nmemb, void* userp){
	std::string* str = static_cast<std::string*>(userp);
	const uint bytes = nmemb*size;
	str->append(static_cast<char*>(buffer), bytes);
	return bytes;
}

class Client {
  s::shared_ptr<CURL>       c;
  s::shared_ptr<curl_slist> headers;
  s::vector<char>           error_buf;
  s::string                 address;
  int                       port;
public:
  Client(const s::string& addr, int port):
    address(addr), port(port){

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

  s::string get(const s::string& route){
    s::string answer;
    s::string url = "http://" + address + route;
    CURLcode r;

    curl_easy_setopt(c.get(), CURLOPT_URL, url.c_str());
    curl_easy_setopt(c.get(), CURLOPT_PORT, port);
    curl_easy_setopt(c.get(), CURLOPT_WRITEDATA, &answer);
    curl_easy_setopt(c.get(), CURLOPT_POST, 0);
    curl_easy_setopt(c.get(), CURLOPT_HTTPHEADER, 0);
    r = curl_easy_perform(c.get());
    if (r){
      s::cout << "error has occurred: " << error_buf.data() << s::endl;
      return s::string();
    }
    return read_response(answer);
  }

  s::string post(const s::string& route, const s::string& post_data){
    s::string answer;
    s::string url = "http://" + address + route;

    curl_easy_setopt(c.get(), CURLOPT_URL, url.c_str());
    curl_easy_setopt(c.get(), CURLOPT_PORT, port);
    curl_easy_setopt(c.get(), CURLOPT_WRITEDATA, &answer);
    curl_easy_setopt(c.get(), CURLOPT_POST, 1);
    curl_easy_setopt(c.get(), CURLOPT_POSTFIELDS, post_data.c_str());
    curl_easy_setopt(c.get(), CURLOPT_POSTFIELDSIZE_LARGE, (curl_off_t)post_data.size());
    curl_easy_setopt(c.get(), CURLOPT_HTTPHEADER, headers.get());

    CURLcode r = curl_easy_perform(c.get());
    if (r){
      s::cout << "error: " << error_buf.data() << s::endl;
      return s::string();
    }
    return read_response(answer);
  }

  s::string read_response(const s::string& answer){
    long response_code;
    CURLcode r = curl_easy_getinfo(c.get(), CURLINFO_RESPONSE_CODE, &response_code);
    if (r){
      s::cout << "error has occurred: " << error_buf.data() << s::endl;
      return s::string();
    }
    s::cout << "response code: " << response_code << s::endl;
    //s::cout << "answer: " << answer << s::endl;
    return answer;
  }
};

int main(){
  Client client("127.0.0.1", 5000);

  //create a new env, list out a working set of environments
  //s::string resp1 = client.post("/v1/envs/", "{\"env_id\":\"CartPole-v0\"}");
  //s::string resp1 = client.post("/v1/envs/", "{\"env_id\":\"MountainCar-v0\"}");
  //s::string resp1 = client.post("/v1/envs/", "{\"env_id\":\"Copy-v0\"}"); //non-discrete action
  //s::string resp1 = client.post("/v1/envs/", "{\"env_id\":\"Blackjack-v0\"}");
  //s::string resp1 = client.post("/v1/envs/", "{\"env_id\":\"NChain-v0\"}");
  //s::string resp1 = client.post("/v1/envs/", "{\"env_id\":\"HotterColder-v0\"}");
  //s::string resp1 = client.post("/v1/envs/", "{\"env_id\":\"Reverse-v0\"}");
  //s::string resp1 = client.post("/v1/envs/", "{\"env_id\":\"Taxi-v3\"}");
  //s::string resp1 = client.post("/v1/envs/", "{\"env_id\":\"Roulette-v0\"}");
  //s::string resp1 = client.post("/v1/envs/", "{\"env_id\":\"FrozenLake8x8-v0\"}");
  s::string resp1 = client.post("/v1/envs/", "{\"env_id\":\"MsPacman-v0\"}");

  //list all env
  s::string resp2 = client.get("/v1/envs/");

  j::Document doc;
  doc.Parse(resp1.c_str());
  s::string instance_id = doc["instance_id"].GetString();

  s::string action_space_request = "/v1/envs/" + instance_id + "/action_space/";
  s::string resp3 = client.get(action_space_request);
  s::cout << "action space: " << resp3 << s::endl;

  s::string monitor_start_request = "/v1/envs/" + instance_id + "/monitor/start/";
  s::string monitor_start_command = "{\"instance_id\":\"" + instance_id + "\",\"directory\":\"./\"}";
  s::string resp7 = client.post(monitor_start_request, monitor_start_command);
  s::cout << "monitor start response: " << resp7 << s::endl;

  //need to call reset before calling step
  s::string reset_request = "/v1/envs/" + instance_id + "/reset/";
  s::string reset_command = "{\"instance_id\":\"" + instance_id + "\"}";
  s::string resp4 = client.post(reset_request, reset_command);
  s::cout << "reset response: " << resp4 << s::endl;

  //does not have observation space
  s::string observation_space_request = "/v1/envs/" + instance_id + "/observation_space/";
  s::string resp5 = client.get(observation_space_request);
  s::cout << "observation space: " << resp5 << s::endl;

  s::string step_request = "/v1/envs/" + instance_id + "/step/";
  s::string step_command = "{\"action\":1}";
  s::string resp6 = client.post(step_request, step_command);
  s::cout << "step response: " << resp6 << s::endl;

  s::string monitor_stop_request = "/v1/envs/" + instance_id + "/monitor/close/";
  s::string resp8 = client.post(monitor_stop_request, "");
  s::cout << "monitor close response: " << resp8 << s::endl;
}
