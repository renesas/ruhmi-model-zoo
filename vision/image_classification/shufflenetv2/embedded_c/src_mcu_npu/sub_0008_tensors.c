#include "sub_0008_tensors.h"

const TensorInfo sub_0008_tensors[] = {
  { "_split_1_command_stream", 0, 13724, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 1, 18480, "MODEL", 0xffffffff },
  { "_split_1_scratch", 2, 94080, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 3, 94080, "FAST_SCRATCH", 0x0 },
  { "model_125_tf_compat_v1_transpose_221_transpose_70190_70441", 4, 37632, "INPUT_TENSOR", 0x9300 },
  { "model_125_tf_compat_v1_transpose_225_transpose_70199", 5, 18816, "OUTPUT_TENSOR", 0x0 },
};

const size_t sub_0008_tensors_count = sizeof(sub_0008_tensors) / sizeof(sub_0008_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0008_address_model_125_tf_compat_v1_transpose_221_transpose_70190_70441 = 0x9300;
const uint32_t sub_0008_address_model_125_tf_compat_v1_transpose_225_transpose_70199 = 0x0;

