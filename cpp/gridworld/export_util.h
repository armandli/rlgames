#ifndef GRIDWORLD_EXPORT_UTIL
#define GRIDWORLD_EXPORT_UTIL

#include <fstream>
#include <vector>

namespace s = std;

namespace gridworld_pt {

void save_loss_array(const char* filename, const s::vector<float>& values){
  s::fstream sm(filename, s::fstream::out);
  sm << "{\"loss\": [\n";
  for (uint i = 0; i < values.size(); ++i){
    if (i != 0)
      sm << ",\n";
    sm << values[i];
  }
  sm << "]}" << s::endl;
}

} // gridworld_pt

#endif//GRIDWORLD_EXPORT_UTIL
