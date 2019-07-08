#include <ctime>
#include <random>
#include <string>
#include <unordered_set>
#include <iostream>
#include <fstream>

#include <type_alias.h>
#include <types.h>

namespace s = std;
namespace R = rlgames;

// generate perfect hash for each player point combination

namespace {
std::default_random_engine& get_default_random_engine(){
  static std::default_random_engine engine(time(0));
  return engine;
}
} //namespace

void generate_file_include(s::fstream& out){
  out << "/* This file is auto-generated */\n";
  out << "#ifndef RLGAMES_ZOBRIST_HASH\n";
  out << "#define RLGAMES_ZOBRIST_HASH\n";
  out << "#include <cassert>\n";
  out << "\n";
  out << "#include <type_alias.h>\n";
  out << "#include <types.h>\n";
  out << "\n";
  out << "namespace rlgames {\n";
  out << "\n";
}

unsigned get_next_value(s::uniform_int_distribution<unsigned>& dist, s::unordered_set<unsigned>& used){
  unsigned value = dist(get_default_random_engine());
  while (used.find(value) != used.end()){
    value = dist(get_default_random_engine());
  }
  return value;
}

void generate_hash_table(s::fstream& out){
  s::uniform_int_distribution<uint> dist(0x10000000U, 0xFFFFFFFFU);
  s::unordered_set<uint> used;

  out << "template <size_t SZ>\n";
  out << "uint zobrist_hash(Player player, Pt pt){\n";
  out << "  assert(player != Player::Unknown);\n";
  out << "\n";
  out << "  uint key = index<SZ>(pt);\n";
  out << "  key |= (uint)player << 16;\n";
  out << "  switch (key){\n";

  out << std::hex;
  for (uint i = 0; i < 19 * 19; ++i){
    uint value = ((uint)R::Player::Black) << 16U;
    value |= i;
    uint hash_value = get_next_value(dist, used);
    used.insert(hash_value);
  out << "    case 0x" << value << "U: return 0x" << hash_value << "U; break;\n";
  }
  for (uint i = 0; i < 19 * 19; ++i){
    uint value = ((uint)R::Player::White) << 16U;
    value |= i;
    uint hash_value = get_next_value(dist, used);
    used.insert(hash_value);
  out << "    case 0x" << value << "U: return 0x" << hash_value << "U; break;\n";
  }
  out << "    default: assert(false);\n";

  out << "  }\n";
  out << "}\n";
}

void generate_file_suffix(s::fstream& out){
  out << "\n";
  out << "constexpr uint EMPTY_BOARD = 0U;\n";
  out << "\n";
  out << "} //rlgames\n";
  out << "#endif//RLGAMES_ZOBRIST_HASH\n";
}

int main(int argc, char* argv[]){
  if (argc != 2){
    s::cout << "Usage: " << argv[0] << " <filename>" << s::endl;
    exit(1);
  }

  const char* filename = argv[1];
  s::fstream output(filename, s::ios_base::out);

  generate_file_include(output);
  generate_hash_table(output);
  generate_file_suffix(output);
  output << s::endl;

  output.close();
}
