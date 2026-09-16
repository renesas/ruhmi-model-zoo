#include "sub_0024_tensors.h"

const TensorInfo sub_0024_tensors[] = {
  { "_split_1_command_stream", 0, 64148, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 1, 229568, "MODEL", 0xffffffff },
  { "_split_1_scratch", 2, 268288, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 3, 268288, "FAST_SCRATCH", 0x0 },
  { "model_135_tf_compat_v1_transpose_335_transpose_70424_70828", 4, 92800, "INPUT_TENSOR", 0x20080 },
  { "model_135_tf_nn_leaky_relu_100_LeakyRelu_70441", 6, 38400, "OUTPUT_TENSOR", 0x0 },
  { "model_135_tf_compat_v1_transpose_340_transpose_70436", 5, 46400, "OUTPUT_TENSOR", 0x9600 },
};

const size_t sub_0024_tensors_count = sizeof(sub_0024_tensors) / sizeof(sub_0024_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0024_address_model_135_tf_compat_v1_transpose_335_transpose_70424_70828 = 0x20080;
const uint32_t sub_0024_address_model_135_tf_nn_leaky_relu_100_LeakyRelu_70441 = 0x0;
const uint32_t sub_0024_address_model_135_tf_compat_v1_transpose_340_transpose_70436 = 0x9600;

