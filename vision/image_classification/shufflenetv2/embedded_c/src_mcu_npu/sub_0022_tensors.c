#include "sub_0022_tensors.h"

const TensorInfo sub_0022_tensors[] = {
  { "_split_1_command_stream", 0, 1220, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 1, 8512, "MODEL", 0xffffffff },
  { "_split_1_scratch", 2, 47040, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 3, 47040, "FAST_SCRATCH", 0x0 },
  { "model_125_tf_reshape_117_Reshape_70298", 5, 18816, "INPUT_TENSOR", 0x4980 },
  { "model_125_tf_compat_v1_transpose_281_transpose_70311", 4, 18816, "OUTPUT_TENSOR", 0x4980 },
};

const size_t sub_0022_tensors_count = sizeof(sub_0022_tensors) / sizeof(sub_0022_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0022_address_model_125_tf_reshape_117_Reshape_70298 = 0x4980;
const uint32_t sub_0022_address_model_125_tf_compat_v1_transpose_281_transpose_70311 = 0x4980;

