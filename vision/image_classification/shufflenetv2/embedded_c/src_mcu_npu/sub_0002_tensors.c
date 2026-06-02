#include "sub_0002_tensors.h"

const TensorInfo sub_0002_tensors[] = {
  { "_split_1_command_stream", 0, 1228, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 1, 3072, "MODEL", 0xffffffff },
  { "_split_1_scratch", 2, 94080, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 3, 94080, "FAST_SCRATCH", 0x0 },
  { "model_125_tf_reshape_97_Reshape_70141", 5, 37632, "INPUT_TENSOR", 0xdc80 },
  { "model_125_tf_compat_v1_transpose_201_transpose_70154", 4, 37632, "OUTPUT_TENSOR", 0x9300 },
};

const size_t sub_0002_tensors_count = sizeof(sub_0002_tensors) / sizeof(sub_0002_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0002_address_model_125_tf_reshape_97_Reshape_70141 = 0xdc80;
const uint32_t sub_0002_address_model_125_tf_compat_v1_transpose_201_transpose_70154 = 0x9300;

