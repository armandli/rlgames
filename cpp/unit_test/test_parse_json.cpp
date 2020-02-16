#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>

#include <string>
#include <iostream>
#include <fstream>

namespace s = std;
namespace j = rapidjson;

static const char* types[] = {
  "Null",
  "False",
  "True",
  "Object",
  "Array",
  "String",
  "Number"
};

void test_read_config(const s::string& filename){
  s::ifstream ifs(filename);
  j::IStreamWrapper isw(ifs);

  j::Document doc;
  doc.ParseStream(isw);

  if (doc.IsArray()){
    s::cout << "doc is array" << s::endl;
  }

  for (auto& v : doc.GetArray()){
    s::cout << types[v.GetType()] << s::endl;
    for (j::Value::ConstMemberIterator it = v.MemberBegin(); it != v.MemberEnd(); ++it){
      s::cout << it->name.GetString() << " " << types[it->value.GetType()] << s::endl;
    }
  }
}

int main(int argc, char* argv[]){
  if (argc != 2){
    s::cout << "Usage: " << argv[0] << " <filename>" << s::endl;
    return 1;
  }

  s::string filename = argv[1];
  test_read_config(filename);
}
