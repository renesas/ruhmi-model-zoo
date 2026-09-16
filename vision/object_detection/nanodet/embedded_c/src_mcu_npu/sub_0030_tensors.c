#include "sub_0030_tensors.h"

const TensorInfo sub_0030_tensors[] = {
  { "_split_1_command_stream", 0, 1188, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 1, 124144, "MODEL", 0xffffffff },
  { "_split_1_scratch", 2, 94400, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 3, 94400, "FAST_SCRATCH", 0x0 },
  { "model_135_tf_compat_v1_transpose_361_transpose_70471_70882", 5, 23200, "INPUT_TENSOR", 0x0 },
  { "model_135_tf_compat_v1_transpose_360_transpose_70473_70876", 4, 23200, "INPUT_TENSOR", 0xb540 },
  { "model_135_tf_compat_v1_transpose_364_transpose_70480", 6, 46400, "OUTPUT_TENSOR", 0xb540 },
};

const size_t sub_0030_tensors_count = sizeof(sub_0030_tensors) / sizeof(sub_0030_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0030_address_model_135_tf_compat_v1_transpose_361_transpose_70471_70882 = 0x0;
const uint32_t sub_0030_address_model_135_tf_compat_v1_transpose_360_transpose_70473_70876 = 0xb540;
const uint32_t sub_0030_address_model_135_tf_compat_v1_transpose_364_transpose_70480 = 0xb540;

