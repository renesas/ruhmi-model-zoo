#include "sub_0000_tensors.h"

const TensorInfo sub_0000_tensors[] = {
  { "_split_1_command_stream", 0, 1628, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 1, 11376, "MODEL", 0xffffffff },
  { "_split_1_scratch", 2, 817280, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 3, 817280, "FAST_SCRATCH", 0x0 },
  { "serving_default_input_0", 5, 150528, "INPUT_TENSOR", 0x0 },
  { "model_125_tf_compat_v1_transpose_193_transpose_70138", 4, 37632, "OUTPUT_TENSOR", 0x9300 },
};

const size_t sub_0000_tensors_count = sizeof(sub_0000_tensors) / sizeof(sub_0000_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0000_address_serving_default_input_0 = 0x0;
const uint32_t sub_0000_address_model_125_tf_compat_v1_transpose_193_transpose_70138 = 0x9300;

