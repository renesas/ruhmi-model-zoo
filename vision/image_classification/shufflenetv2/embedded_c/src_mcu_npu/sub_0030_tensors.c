#include "sub_0030_tensors.h"

const TensorInfo sub_0030_tensors[] = {
  { "_split_1_command_stream", 0, 1228, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 1, 24112, "MODEL", 0xffffffff },
  { "_split_1_scratch", 2, 23520, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 3, 23520, "FAST_SCRATCH", 0x0 },
  { "model_125_tf_reshape_125_Reshape_70359", 5, 9408, "INPUT_TENSOR", 0x24c0 },
  { "model_125_tf_compat_v1_transpose_313_transpose_70372", 4, 9408, "OUTPUT_TENSOR", 0x24c0 },
};

const size_t sub_0030_tensors_count = sizeof(sub_0030_tensors) / sizeof(sub_0030_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0030_address_model_125_tf_reshape_125_Reshape_70359 = 0x24c0;
const uint32_t sub_0030_address_model_125_tf_compat_v1_transpose_313_transpose_70372 = 0x24c0;

