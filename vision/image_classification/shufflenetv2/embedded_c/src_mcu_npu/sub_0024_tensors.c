#include "sub_0024_tensors.h"

const TensorInfo sub_0024_tensors[] = {
  { "_split_1_command_stream", 0, 25980, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 1, 44800, "MODEL", 0xffffffff },
  { "_split_1_scratch", 2, 48880, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 3, 48880, "FAST_SCRATCH", 0x0 },
  { "model_125_tf_compat_v1_transpose_285_transpose_70315_70579", 4, 18816, "INPUT_TENSOR", 0x4980 },
  { "model_125_tf_compat_v1_transpose_289_transpose_70324", 5, 9408, "OUTPUT_TENSOR", 0x0 },
};

const size_t sub_0024_tensors_count = sizeof(sub_0024_tensors) / sizeof(sub_0024_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0024_address_model_125_tf_compat_v1_transpose_285_transpose_70315_70579 = 0x4980;
const uint32_t sub_0024_address_model_125_tf_compat_v1_transpose_289_transpose_70324 = 0x0;

