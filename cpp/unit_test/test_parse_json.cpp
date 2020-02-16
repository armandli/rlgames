#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/writer.h>

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

void test_write_config(const s::string& filename){
  j::Document doc;
  doc.SetObject();
  j::Document::AllocatorType& allocator = doc.GetAllocator();

  j::Value key("some_key");
  j::Value value(j::kArrayType);
  for (int i = 0; i < 10; ++i)
    value.PushBack(j::Value().SetFloat(i * 10.F), allocator);
  doc.AddMember(key, value, allocator);

  s::ofstream ofs(filename);
  j::OStreamWrapper osw(ofs);
  j::Writer<j::OStreamWrapper> writer(osw);
  doc.Accept(writer);
}

int main(int argc, char* argv[]){
  if (argc != 3){
    s::cout << "Usage: " << argv[0] << " <input filename> <output filename>" << s::endl;
    return 1;
  }

  s::string input_filename = argv[1];
  test_read_config(input_filename);

  s::string output_filename = argv[2];
  test_write_config(output_filename);
}
