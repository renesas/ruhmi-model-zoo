#include "sub_0000_tensors.h"

const TensorInfo sub_0000_tensors[] = {
  { "_split_1_command_stream", 0, 42540, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 1, 22112, "MODEL", 0xffffffff },
  { "_split_1_scratch", 2, 1157248, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 3, 1157248, "FAST_SCRATCH", 0x0 },
  { "serving_default_data_0", 5, 307200, "INPUT_TENSOR", 0xcd080 },
  { "model_135_tf_compat_v1_transpose_242_transpose_70262", 4, 185600, "OUTPUT_TENSOR", 0x2d500 },
};

const size_t sub_0000_tensors_count = sizeof(sub_0000_tensors) / sizeof(sub_0000_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0000_address_serving_default_data_0 = 0xcd080;
const uint32_t sub_0000_address_model_135_tf_compat_v1_transpose_242_transpose_70262 = 0x2d500;

