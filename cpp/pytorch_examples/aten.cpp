#include <iostream>

#include <ATen/ATen.h>

using namespace std;

int main(){
  at::Tensor a = at::ones({2, 2}, at::kInt);
  at::Tensor b = at::randn({2, 2});
  auto c = a + b.to(at::kInt);

  std::cout << c << std::endl;
}
