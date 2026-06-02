#include "sub_0006_tensors.h"

const TensorInfo sub_0006_tensors[] = {
  { "_split_1_command_stream", 0, 1228, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 1, 3008, "MODEL", 0xffffffff },
  { "_split_1_scratch", 2, 94080, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 3, 94080, "FAST_SCRATCH", 0x0 },
  { "model_125_tf_reshape_101_Reshape_70173", 5, 37632, "INPUT_TENSOR", 0xdc80 },
  { "model_125_tf_compat_v1_transpose_217_transpose_70186", 4, 37632, "OUTPUT_TENSOR", 0x9300 },
};

const size_t sub_0006_tensors_count = sizeof(sub_0006_tensors) / sizeof(sub_0006_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0006_address_model_125_tf_reshape_101_Reshape_70173 = 0xdc80;
const uint32_t sub_0006_address_model_125_tf_compat_v1_transpose_217_transpose_70186 = 0x9300;

