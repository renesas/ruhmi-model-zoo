#include "sub_0032_tensors.h"

const TensorInfo sub_0032_tensors[] = {
  { "_split_1_command_stream", 1, 976, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 2, 1174432, "MODEL", 0xffffffff },
  { "_split_1_scratch", 3, 59584, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 4, 59584, "FAST_SCRATCH", 0x0 },
  { "model_125_tf_compat_v1_transpose_317_transpose_70376_70645", 5, 9408, "INPUT_TENSOR", 0x24c0 },
  { "StatefulPartitionedCall_0_70379", 0, 1000, "OUTPUT_TENSOR", 0x0 },
};

const size_t sub_0032_tensors_count = sizeof(sub_0032_tensors) / sizeof(sub_0032_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0032_address_model_125_tf_compat_v1_transpose_317_transpose_70376_70645 = 0x24c0;
const uint32_t sub_0032_address_StatefulPartitionedCall_0_70379 = 0x0;

