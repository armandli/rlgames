#include <iostream>

#include <torch/torch.h>

//test out how softmax operator in pytorch works

namespace s = std;
namespace t = torch;

class TestModelImpl : public t::nn::Module {
  t::nn::Linear l1, l2;
public:
  TestModelImpl(unsigned input, unsigned hidden, unsigned output):
    l1(register_module("l1", t::nn::Linear(input, hidden))),
    l2(register_module("l2", t::nn::Linear(hidden, output)))
  {}

  t::Tensor forward(t::Tensor x){
    x = t::relu(l1(x));
    x = t::softmax(l2(x), -1);
    return x;
  }
};
TORCH_MODULE(TestModel);

void test_softmax(){
  TestModel model(4, 8, 4);

  float input1_array[] = {1., 2., 3., 4.};
  t::Tensor input1 = t::from_blob(input1_array, {4});
  t::Tensor output1 = model->forward(input1);

  float input2_array[] = {1., 2., 3., 4., 5., 6., 7., 8.};
  t::Tensor input2 = t::from_blob(input2_array, {2, 4});
  t::Tensor output2 = model->forward(input2);

  s::cout << "Output1:\n";
  s::cout << output1 << s::endl;
  s::cout << "Output2:\n";
  s::cout << output2 << s::endl;
  s::cout << "Input2:\n";
  s::cout << input2 << s::endl;
}

void test_index_select(){
  float src_data[] = {
    0, 1, 2, 3,
    4, 5, 6, 7,
    8, 9, 10, 11,
    12, 13, 14, 15,
  };
  t::Tensor src = t::from_blob(src_data, {4, 4});
  long index_data[] = {1, 2};
  t::Tensor index = t::from_blob(index_data, {2}, t::kLong);
  t::Tensor r = t::index_select(src, 0, index);

  s::cout << "Index Select" << s::endl;
  s::cout << r << s::endl;
}

void test_index_select2(){
  float src_data[] = {
    0, 1, 2, 3,
    4, 5, 6, 7,
    8, 9, 10, 11,
    12, 13, 14, 15,
    16, 17, 18, 19,
    20, 21, 22, 23,
    24, 25, 26, 27,
    28, 29, 30, 31,
  };
  t::Tensor src = t::from_blob(src_data, {8, 4});
  long index_data[] = {1, 2, 0, 3, 0, 0, 3, 2};
  t::Tensor index = t::from_blob(index_data, {8}, t::kLong);
  t::Tensor r = t::index_select(src, 1, index);
  r = r.diagonal();

  s::cout << "Index Select 2" << s::endl;
  s::cout << r << s::endl;
}


void test_masked_select(){
  float src_data[] = {
    0, 1, 2, 3,
    4, 5, 6, 7,
    8, 9, 10, 11,
    12, 13, 14, 15,
  };
  t::Tensor src = t::from_blob(src_data, {4, 4});
  bool mask_data[] = {
    0, 0, 0, 1,
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
  };
  t::Tensor mask = t::from_blob(mask_data, {4, 4}, t::kBool);
  t::Tensor r = t::masked_select(src, mask);

  s::cout << "Masked select" << s::endl;
  s::cout << r << s::endl;
}

void test_max(){
  float src_data[] = {
    0, 1, 2, 3,
    4, 5, 6, 7,
    8, 9, 10, 11,
    12, 13, 14, 15,
  };
  t::Tensor src = t::from_blob(src_data, {4, 4});
  t::Tensor r = t::max_values(src, 1);

  s::cout << "Max:" << s::endl;
  s::cout << r << s::endl;
}

void test_logical_not(){
  bool src_data[] = {
    true, false, false, false,
    false, true, false, false,
    false, false, true, false,
    false, false, false, true,
  };
  t::Tensor src = t::from_blob(src_data, {4, 4}, t::kBool);
  t::Tensor r = src.logical_not();

  s::cout << "Logical Not:" << s::endl;
  s::cout << r << s::endl;

  float v_data[] = {
    0, 1, 2, 3,
    4, 5, 6, 7,
    8, 9, 10, 11,
    12, 13, 14, 15,
  };
  t::Tensor v = t::from_blob(v_data, {4, 4});
  r = r * v;

  s::cout << r << s::endl;
}

void test_argmax(){
  float src_data[] = {
    0, 1, 2, 3,
    4, 5, 6, 7,
    8, 9, 10, 11,
    12, 13, 14, 15,
  };
  t::Tensor src = t::from_blob(src_data, {4, 4});
  t::Tensor r = t::argmax(src, 1);

  s::cout << "ArgMax:" << s::endl;
  s::cout << r << s::endl;
}

int main(){
  test_softmax();
  test_index_select();
  test_index_select2();
  test_masked_select();
  test_max();
  test_logical_not();
  test_argmax();
}
