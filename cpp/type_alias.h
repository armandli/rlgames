#include <cstdint>
#include <initializer_list>

using ubyte = uint8_t;
using uint = uint32_t;
using sint = int32_t;
using udyte = uint16_t;
using sint64 = int64_t;
using uint64 = uint64_t;

template <typename T>
using init_list = std::initializer_list<T>;
