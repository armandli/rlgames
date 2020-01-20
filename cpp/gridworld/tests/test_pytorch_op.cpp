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

void test_dimension_reshape(){
  float src_data[] = {
    0, 1, 2, 3,
    5, 5, 5, 5,
    10, 10, 20, 10,
    12, 13, 14, 15,
  };
  t::Tensor src = t::from_blob(src_data, {4, 4});
  t::Tensor r = src.reshape({4, 2, -1});

  s::cout << "Reshape to create a new dimension" << s::endl;
  s::cout << r << s::endl;

  r = t::softmax(r, -1);

  s::cout << "After softmax" << s::endl;
  s::cout << r << s::endl;
  s::cout << "Diemsnion " << r.dim() << s::endl;
  s::cout << "Size of first dimension " << r.size(0) << s::endl;
}

void test_split(){
  float src_data[] = {
    0, 1, 2, 3,
    4, 5, 6, 7,
    8, 9, 10, 11,
    12, 13, 14, 15,
  };
  t::Tensor src = t::from_blob(src_data, {4, 4});
  s::vector<t::Tensor> r = src.split(2, -1);

  s::cout << "Split Tensor Before softmax" << s::endl;
  for (uint i = 0; i < r.size(); ++i)
    s::cout << r[i] << s::endl;

  for (uint i = 0 ; i < r.size(); ++i)
    r[i] = t::softmax(r[i], -1);

  s::cout << "After softmax" << s::endl;
  for (uint i = 0; i < r.size(); ++i)
    s::cout << r[i] << s::endl;
}

void test_softmax2(){
  float src_data[] = {
    0, 1, 2, 3,
    4, 5, 6, 7,
    8, 9, 10, 11,
    12, 13, 14, 15,
  };
  t::Tensor src = t::from_blob(src_data, {4, 4});
  t::Tensor r = t::softmax(src, -1);

  s::cout << "Test Softmax" << s::endl;
  s::cout << r << s::endl;
}

void test_row_multiply(){
  float src_data[] = {
    0, 1, 2, 3,
    4, 5, 6, 7,
    8, 9, 10, 11,
    12, 13, 14, 15,
  };
  t::Tensor src = t::from_blob(src_data, {4, 4});
  src = src.reshape({4, 2, 2});

  float src2_data[] = {
    0, 1
  };
  t::Tensor vec = t::from_blob(src2_data, {2});

  t::Tensor r = src * vec;

  s::cout << "Test Uneven row multiply" << s::endl;
  s::cout << r << s::endl;
}

void test_index_select_tensor(){
  float src_data[] = {
    0, 1, 2, 3,
    4, 5, 6, 7,
    8, 9, 10, 11,
    12, 13, 14, 15,
  };
  t::Tensor src = t::from_blob(src_data, {4, 4});
  src = src.reshape({4, 2, 2});

  long index_data[] = {0, 1, 1, 0};
  t::Tensor index = t::from_blob(index_data, {4}, t::kLong);
  s::cout << "Index Select on Tensor" << s::endl;

  s::cout << "original source: " << s::endl;
  s::cout << src << s::endl;

  t::Tensor r = t::index_select(src, 1, index);

  s::cout << "result" << s::endl;
  s::cout << r << s::endl;
}

void test_gather_and_narrow_and_squeeze(){
  float src_data[] = {
    0, 1, 2, 3,
    4, 5, 6, 7,
    8, 9, 10, 11,
    12, 13, 14, 15,
  };
  t::Tensor src = t::from_blob(src_data, {4, 4});
  src = src.reshape({4, 2, 2});

  long index_data[] = {
    1, 1, 1, 1,
    0, 0, 0, 0,
    1, 1, 1, 1,
    0, 0, 0, 0};
  t::Tensor index = t::from_blob(index_data, {4, 2, 2}, t::kLong);
  t::Tensor r = t::gather(src, 1, index);

  s::cout << "Gather" << s::endl;
  s::cout << r << s::endl;

  r = t::narrow(r, 1, 0, 1);
  r = t::squeeze(r);

  s::cout << "Narrow and Squeeze" << s::endl;
  s::cout << r << s::endl;
}

void test_gather2(){
  float src_data[] = {
    0, 1, 2, 3,
    4, 5, 6, 7,
    8, 9, 10, 11,
    12, 13, 14, 15,
  };
  t::Tensor src = t::from_blob(src_data, {4, 4});
  src = src.reshape({4, 2, 2});

  long index_data[] = {
    1, 1,
    0, 0,
    1, 1,
    0, 0,
  };
  t::Tensor index = t::from_blob(index_data, {4, 1, 2}, t::kLong);
  t::Tensor r = t::gather(src, 1, index);

  s::cout << "Gather2" << s::endl;
  s::cout << r << s::endl;
}

void test_gather3(){
  float src_data[] = {
    0, 1, 2, 3,
    4, 5, 6, 7,
    8, 9, 10, 11,
    12, 13, 14, 15,
  };
  t::Tensor src = t::from_blob(src_data, {4, 4});
  src = src.reshape({2, 4, 2});

  s::cout << "Gather3" << s::endl;
  s::cout << "src" << s::endl;
  s::cout << src << s::endl;

  long index_data[] = {
    0, 0,
    1, 1,
    0, 0,
    1, 1,
  };
  t::Tensor index = t::from_blob(index_data, {1, 4, 2}, t::kLong);
  t::Tensor r = t::gather(src, 0, index);

  s::cout << "result" << s::endl;
  s::cout << r << s::endl;
}

void test_matrix_vector_multiply(){
  float src_data[] = {
    0, 1, 2, 3,
    4, 5, 6, 7,
    8, 9, 10, 11,
    12, 13, 14, 15,
  };
  t::Tensor src = t::from_blob(src_data, {4, 4});

  float src2_data[] = {
    0.1, 0.1, 0.1, 0.1
  };
  t::Tensor vec = t::from_blob(src2_data, {4});

  t::Tensor r = (src * vec).sum(-1);

  s::cout << "Matrix vector multiply" << s::endl;
  s::cout << r << s::endl;
}

void test_matrix_cell_multiply(){
  float src_data[] = {
    0, 1, 2, 3,
    4, 5, 6, 7,
    8, 9, 10, 11,
    12, 13, 14, 15,
  };
  t::Tensor src1 = t::from_blob(src_data, {4, 4});
  t::Tensor src2 = t::from_blob(src_data, {4, 4});

  t::Tensor r = src1 * src2;

  s::cout << "Matrix Matrix cell multiply" << s::endl;
  s::cout << r << s::endl;
}

void test_cat(){
  float src_data[] = {
    0, 1, 2, 3,
    4, 5, 6, 7,
    8, 9, 10, 11,
    12, 13, 14, 15,
  };
  t::Tensor src1 = t::from_blob(src_data, {4, 4});
  src1 = src1.reshape({1, 4, 4,});

  float src2_data[] = {
    16, 17, 18, 19,
    20, 21, 22, 23,
    24, 25, 26, 27,
    28, 29, 30, 31,
  };
  t::Tensor src2 = t::from_blob(src2_data, {4, 4});
  src2 = src2.reshape({1, 4, 4});

  t::Tensor r = t::cat({src1, src2}, 0);

  s::cout << "Concatenate matrixes into tensor" << s::endl;
  s::cout << r << s::endl;

  long select_data[] = {
    0, 1, 0, 1,
  };
  t::Tensor select = t::from_blob(select_data, {4}, t::kLong);
  select = select.repeat_interleave(4);
  select = select.reshape({1, 4, 4});

  r = t::squeeze(t::gather(r, 0, select));

  s::cout << "After doing gather" << s::endl;
  s::cout << r << s::endl;
}

void test_logical_not2(){
  long select_data[] = {
    0, 1, 0, 1,
  };
  t::Tensor select = t::from_blob(select_data, {4}, t::kLong);
  select = select.logical_not();

  s::cout << "Logical Not on Long Vector" << s::endl;
  s::cout << select << s::endl;
}

void test_from_blob_selection(){
  float src_data[] = {
    0, 1, 2, 3,
    4, 5, 6, 7,
    8, 9, 10, 11,
    12, 13, 14, 15,
  };
  t::Tensor src1 = t::from_blob(src_data, {12});

  s::cout << "test from blob" << s::endl;
  s::cout << src1 << s::endl;
}

void test_slice(){
  float src_data[] = {
    0, 1, 2, 3,
    4, 5, 6, 7,
    8, 9, 10, 11,
    12, 13, 14, 15,
  };
  t::Tensor src1 = t::from_blob(src_data, {16});
  t::Tensor r = src1.slice(0, 0, 9);

  s::cout << "slicing" << s::endl;
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
  test_dimension_reshape();
  test_split();
  test_softmax2();
  test_row_multiply();
  test_index_select_tensor();
  test_gather_and_narrow_and_squeeze();
  test_gather2();
  test_gather3();
  test_matrix_vector_multiply();
  test_matrix_cell_multiply();
  test_cat();
  test_logical_not2();
  test_from_blob_selection();
  test_slice();
}
