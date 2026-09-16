#include "sub_0008_tensors.h"

const TensorInfo sub_0008_tensors[] = {
  { "_split_1_command_stream", 0, 36816, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 1, 84096, "MODEL", 0xffffffff },
  { "_split_1_scratch", 2, 596624, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 3, 596624, "FAST_SCRATCH", 0x0 },
  { "model_135_tf_compat_v1_transpose_270_transpose_70308_70690", 4, 185600, "INPUT_TENSOR", 0x52d00 },
  { "model_135_tf_nn_leaky_relu_82_LeakyRelu_70325", 6, 153600, "OUTPUT_TENSOR", 0x0 },
  { "model_135_tf_compat_v1_transpose_275_transpose_70320", 5, 92800, "OUTPUT_TENSOR", 0x25800 },
};

const size_t sub_0008_tensors_count = sizeof(sub_0008_tensors) / sizeof(sub_0008_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0008_address_model_135_tf_compat_v1_transpose_270_transpose_70308_70690 = 0x52d00;
const uint32_t sub_0008_address_model_135_tf_nn_leaky_relu_82_LeakyRelu_70325 = 0x0;
const uint32_t sub_0008_address_model_135_tf_compat_v1_transpose_275_transpose_70320 = 0x25800;

