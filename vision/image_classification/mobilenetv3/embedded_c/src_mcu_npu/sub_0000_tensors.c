#include "sub_0000_tensors.h"

const TensorInfo sub_0000_tensors[] = {
  { "_split_1_command_stream", 1, 6792, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 2, 2143760, "MODEL", 0xffffffff },
  { "_split_1_scratch", 3, 258048, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 4, 258048, "FAST_SCRATCH", 0x0 },
  { "serving_default_keras_tensor_110_0", 5, 110592, "INPUT_TENSOR", 0x0 },
  { "StatefulPartitionedCall_1_0_70122", 0, 1000, "OUTPUT_TENSOR", 0x0 },
};

const size_t sub_0000_tensors_count = sizeof(sub_0000_tensors) / sizeof(sub_0000_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0000_address_serving_default_keras_tensor_110_0 = 0x0;
const uint32_t sub_0000_address_StatefulPartitionedCall_1_0_70122 = 0x0;

