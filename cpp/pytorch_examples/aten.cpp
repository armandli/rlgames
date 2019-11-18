#include <iostream>

#include <ATen/ATen.h>

using namespace std;

int main(){
  at::Tensor a = at::ones({2, 2}, at::kInt);
  at::Tensor b = at::randn({2, 2});
  auto c = a + b.to(at::kInt);

  std::cout << c << std::endl;

  std::cout << a[0][0] << std::endl;

  a[0][0] = 1000;

  std::cout << a[0][0] << std::endl;

  float data[] = {1., 2., 3., 4., 5., 6.};
  at::Tensor d = at::from_blob(data, {3, 2});

  std::cout << d << std::endl;

  d[1][1] = 10.;

  std::cout << d << std::endl;
}
